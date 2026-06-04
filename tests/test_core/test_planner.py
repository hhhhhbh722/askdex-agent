# -*- coding: utf-8 -*-
"""PlannerAgent 测试：计划生成、执行、重规划。"""

from __future__ import annotations

import json

import pytest

from app.core.agent.planner import (
    PlannerAgent,
    SubTask,
    _extract_json_object,
    _parse_subtasks,
)
from tests.conftest import MockLLM, MockToolInvoker


# ============================================================================
# 纯函数测试
# ============================================================================


class TestExtractJsonObject:
    """JSON 提取函数测试。"""

    def test_extract_plain_json(self) -> None:
        text = '{"subtasks": [{"id": "t1", "title": "Test"}]}'
        result = _extract_json_object(text)
        assert result["subtasks"][0]["id"] == "t1"

    def test_extract_json_from_text(self) -> None:
        text = 'Here is the plan:\n```json\n{"subtasks": [{"id": "t2"}]}\n```'
        result = _extract_json_object(text)
        assert result["subtasks"][0]["id"] == "t2"

    def test_extract_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="无法从模型输出中解析 JSON"):
            _extract_json_object("Not JSON at all, nowhere.")


class TestParseSubtasks:
    """子任务解析测试。"""

    def test_parse_normal(self) -> None:
        data = {
            "subtasks": [
                {"id": "t1", "title": "Search", "description": "Search the web",
                 "action_type": "tool", "tool_name": "search"},
                {"id": "t2", "title": "Analyze", "description": "Analyze results",
                 "action_type": "reasoning", "tool_name": None},
            ]
        }
        tasks = _parse_subtasks(data)
        assert len(tasks) == 2
        assert tasks[0].id == "t1"
        assert tasks[0].action_type == "tool"
        assert tasks[1].action_type == "reasoning"

    def test_parse_empty(self) -> None:
        tasks = _parse_subtasks({})
        assert tasks == []

    def test_parse_missing_fields(self) -> None:
        data = {"subtasks": [{"id": "t1"}]}
        tasks = _parse_subtasks(data)
        assert len(tasks) == 1
        assert tasks[0].title == ""
        assert tasks[0].action_type == "reasoning"  # default

    def test_action_type_normalized(self) -> None:
        data = {"subtasks": [{"id": "t1", "action_type": "TOOL"}]}
        tasks = _parse_subtasks(data)
        assert tasks[0].action_type == "tool"


# ============================================================================
# PlannerAgent 测试（使用 Protocol 模拟）
# ============================================================================


_VALID_PLAN_JSON = json.dumps({
    "subtasks": [
        {"id": "t1", "title": "Search info", "description": "Search for AI",
         "action_type": "tool", "tool_name": "search", "tool_args_hint": "AI agents"},
        {"id": "t2", "title": "Analyze", "description": "Analyze search results",
         "action_type": "reasoning", "tool_name": None, "tool_args_hint": None},
    ]
})


@pytest.fixture
def planner(mock_llm: MockLLM, mock_tools: MockToolInvoker) -> PlannerAgent:
    return PlannerAgent(llm=mock_llm, tools=mock_tools, memory=None, max_replan_attempts=2)


class TestPlannerPlan:
    """计划生成阶段测试。"""

    async def test_plan_success(self, planner: PlannerAgent, mock_llm: MockLLM) -> None:
        mock_llm.add_response(_VALID_PLAN_JSON)
        plan = await planner.plan("Tell me about AI agents")
        assert len(plan) == 2
        assert plan[0].id == "t1"
        assert plan[0].action_type == "tool"

    async def test_plan_llm_failure_fallback(self, planner: PlannerAgent, mock_llm: MockLLM) -> None:
        async def raise_error(messages, **kwargs) -> str:
            raise RuntimeError("LLM unavailable")

        mock_llm.acomplete = raise_error  # type: ignore[assignment]
        plan = await planner.plan("Some query")
        # 应降级为单个回退任务
        assert len(plan) == 1
        assert plan[0].id == "fallback_1"

    async def test_plan_invalid_json_fallback(self, planner: PlannerAgent, mock_llm: MockLLM) -> None:
        mock_llm.add_response("Not a valid JSON response at all")
        plan = await planner.plan("query")
        assert len(plan) == 1
        assert plan[0].action_type == "reasoning"


class TestPlannerExecute:
    """计划执行阶段测试。"""

    async def test_execute_tool_task(
        self, planner: PlannerAgent, mock_llm: MockLLM, mock_tools: MockToolInvoker
    ) -> None:
        plan = [
            SubTask(id="t1", title="Search", description="Search web",
                    action_type="tool", tool_name="search"),
        ]
        mock_tools.results["search"] = "Found results about AI"
        # 汇总需 LLM
        mock_llm.add_response("Summary: AI is artificial intelligence.")

        result = await planner.execute(plan, "What is AI?", "sess-1", tool_names=["search"])
        assert result.success is True
        assert "AI" in result.final_answer

    async def test_execute_reasoning_task(
        self, planner: PlannerAgent, mock_llm: MockLLM
    ) -> None:
        plan = [
            SubTask(id="t1", title="Think", description="Reason about the question",
                    action_type="reasoning"),
        ]
        # 推理子任务需要 LLM
        mock_llm.add_response("Analyzed context result")
        # 汇总
        mock_llm.add_response("Final answer after analysis.")

        result = await planner.execute(plan, "Think deeply", "sess-1")
        assert result.success is True

    async def test_execute_tool_not_allowed(
        self, planner: PlannerAgent, mock_tools: MockToolInvoker
    ) -> None:
        plan = [
            SubTask(id="t1", title="Search", description="Search",
                    action_type="tool", tool_name="search"),
        ]
        result = await planner.execute(
            plan, "query", "sess-1", tool_names=["calculator"]  # search not allowed
        )
        assert result.success is False

    async def test_execute_subtask_failure(
        self, planner: PlannerAgent, mock_tools: MockToolInvoker
    ) -> None:
        async def fail_invoke(name, arguments):
            raise RuntimeError("Tool error")

        mock_tools.invoke = fail_invoke  # type: ignore[assignment]
        plan = [
            SubTask(id="t1", title="Search", description="Search",
                    action_type="tool", tool_name="search"),
        ]
        result = await planner.execute(plan, "query", "sess-1")
        assert result.success is False
        assert "Tool error" in (result.error or "")


class TestPlannerReplan:
    """重规划阶段测试。"""

    async def test_replan_success(self, planner: PlannerAgent, mock_llm: MockLLM) -> None:
        results = [{"subtask_id": "t1", "status": "error", "error": "Tool failed"}]
        mock_llm.add_response(json.dumps({
            "subtasks": [{"id": "t2", "title": "Fallback", "description": "Try differently",
                          "action_type": "reasoning"}],
            "notes": "Tool failed, switching to reasoning"
        }))
        new_plan = await planner.replan(
            plan=[SubTask(id="t1", title="Search", description="Search",
                          action_type="tool", tool_name="search")],
            results=results,
            error="Tool failed",
        )
        assert len(new_plan) == 1
        assert new_plan[0].id == "t2"

    async def test_replan_failure_fallback(self, planner: PlannerAgent, mock_llm: MockLLM) -> None:
        async def raise_error(messages, **kwargs):
            raise RuntimeError("Replan failed")
        mock_llm.acomplete = raise_error  # type: ignore[assignment]

        new_plan = await planner.replan(
            plan=[],
            results=[],
            error="Some error",
        )
        assert len(new_plan) == 1
        assert new_plan[0].id == "replan_fallback"


class TestRunWithReplan:
    """run_with_replan 整合测试。"""

    async def test_success_first_try(
        self, planner: PlannerAgent, mock_llm: MockLLM, mock_tools: MockToolInvoker
    ) -> None:
        # plan 调用
        mock_llm.add_response(_VALID_PLAN_JSON)
        # execute reasoning + summary 调用
        mock_llm.add_response("Analyzed.")  # reasoning
        mock_llm.add_response("Final summarized answer.")  # summary
        mock_tools.results["search"] = "Search results."

        result = await planner.run_with_replan("What is AI?", "sess-1", tool_names=["search"])
        assert result.success is True
        assert "Final summarized answer" in result.final_answer

    async def test_all_attempts_fail(
        self, planner: PlannerAgent, mock_llm: MockLLM
    ) -> None:
        # 每次都返回无效规划 JSON，plan 会 fallback 为 reasoning task
        mock_llm.add_response("not json")  # plan attempt → fallback
        # execute: reasoning task 需要 LLM
        mock_llm.add_response("exec result.")  # execute reasoning
        mock_llm.add_response("summary.")  # summary

        result = await planner.run_with_replan("query", "sess-1")
        # fallback 计划中的 reasoning task 应成功
        assert result.success is True
