# -*- coding: utf-8 -*-
"""ReActAgent 测试：解析函数 + 完整 Agent 循环。"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.core.agent.react_agent import (
    AgentResult,
    ReActAgent,
    _parse_react_step,
    build_react_user_prompt,
)
from tests.conftest import MockLLM, MockMemoryLike, MockToolInvoker


# ============================================================================
# _parse_react_step 纯函数测试
# ============================================================================


class TestParseReactStep:
    """ReAct 输出解析测试。"""

    def test_parse_final_answer(self) -> None:
        text = "Thought: I have enough info.\nFinal Answer: The answer is 42."
        result = _parse_react_step(text)
        assert result["done"] is True
        assert result["final_answer"] == "The answer is 42."
        assert result["thought"] == "I have enough info."

    def test_parse_action_with_valid_json(self) -> None:
        text = 'Thought: Need to calculate.\nAction: calculator\nAction Input: {"expression": "2+2"}'
        result = _parse_react_step(text)
        assert result["action"] == "calculator"
        assert result["action_input"] == {"expression": "2+2"}

    def test_parse_malformed_json(self) -> None:
        text = "Thought: Let me search.\nAction: search\nAction Input: {not valid json}"
        result = _parse_react_step(text)
        assert result["action"] == "search"
        assert "parse_error" in result

    def test_parse_no_thought(self) -> None:
        text = "Action: search\nAction Input: {\"query\": \"AI\"}"
        result = _parse_react_step(text)
        assert result["action"] == "search"

    def test_parse_neither_action_nor_final(self) -> None:
        text = "Just some random text without proper format."
        result = _parse_react_step(text)
        assert "action" not in result
        assert "final_answer" not in result

    def test_parse_final_answer_with_unicode(self) -> None:
        text = "Thought: 已有足够信息。\nFinal Answer: 答案是：人工智能代理"
        result = _parse_react_step(text)
        assert result["done"] is True
        assert "人工智能代理" in result["final_answer"]

    def test_parse_nested_json_action_input(self) -> None:
        text = (
            "Thought: Complex query.\n"
            'Action: database\n'
            'Action Input: {"sql": "SELECT * FROM users", "params": {"limit": 10}}'
        )
        result = _parse_react_step(text)
        assert result["action"] == "database"
        assert result["action_input"]["sql"] == "SELECT * FROM users"
        assert result["action_input"]["params"]["limit"] == 10


# ============================================================================
# build_react_user_prompt 纯函数测试
# ============================================================================


class TestBuildReactUserPrompt:
    """提示构造测试。"""

    def test_build_prompt_with_tools(self) -> None:
        prompt = build_react_user_prompt(
            query="What is AI?",
            tool_descriptions="- search: web search\n- calculator: math",
            history_block="（尚无）",
        )
        assert "What is AI?" in prompt
        assert "search" in prompt
        assert "calculator" in prompt

    def test_build_prompt_no_tools(self) -> None:
        prompt = build_react_user_prompt(
            query="Hello",
            tool_descriptions="（无外部工具，请直接 Final Answer）",
            history_block="（尚无）",
        )
        assert "Hello" in prompt
        assert "Final Answer" in prompt

    def test_build_prompt_with_history(self) -> None:
        history = "Step 1\nAction: search\nObservation: Found results"
        prompt = build_react_user_prompt(
            query="Continue",
            tool_descriptions="- search",
            history_block=history,
        )
        assert "Continue" in prompt
        assert "search" in prompt
        assert "Step 1" in prompt

    def test_build_prompt_empty_history(self) -> None:
        prompt = build_react_user_prompt(
            query="Start",
            tool_descriptions="- search",
            history_block="（尚无）",
        )
        assert "尚无" in prompt


# ============================================================================
# ReActAgent 完整循环测试（使用 Protocol 模拟）
# ============================================================================


class TestReActAgentLoop:
    """ReAct Agent 完整执行循环测试。"""

    @pytest.fixture
    def agent(self, mock_llm: MockLLM, mock_tools: MockToolInvoker) -> ReActAgent:
        return ReActAgent(
            llm=mock_llm,
            tools=mock_tools,
            memory=None,
            max_steps=5,
        )

    async def test_successful_run_final_answer(
        self, agent: ReActAgent, mock_llm: MockLLM
    ) -> None:
        """LLM 直接返回 Final Answer 的成功路径。"""
        mock_llm.add_response(
            "Thought: I can answer directly.\nFinal Answer: AI Agent is an autonomous system."
        )
        result = await agent.run(
            query="What is an AI agent?",
            context={"tool_names": ["search"]},
        )
        assert result.success is True
        assert "AI Agent" in result.final_answer

    async def test_tool_invocation_then_final(
        self,
        agent: ReActAgent,
        mock_llm: MockLLM,
        mock_tools: MockToolInvoker,
    ) -> None:
        """先调用工具，再给出最终答案。"""
        # 第一步：调用工具
        mock_llm.add_response(
            'Thought: I need to search.\nAction: search\nAction Input: {"query": "AI agents"}'
        )
        mock_tools.results["search"] = "AI agents are autonomous software entities."
        # 第二步：最终答案
        mock_llm.add_response(
            "Thought: Got the info.\nFinal Answer: Based on search, AI agents are..."
        )
        result = await agent.run(
            query="Tell me about AI agents",
            context={"tool_names": ["search"]},
        )
        assert result.success is True
        assert len(result.steps) == 2
        # 第一步应该包含观察结果
        step1 = result.steps[0]
        assert step1["action"] == "search"
        assert "AI agents are" in step1["observation"]

    async def test_max_steps_exceeded(
        self,
        agent: ReActAgent,
        mock_llm: MockLLM,
    ) -> None:
        """超过最大步数限制应返回失败。"""
        # 永远返回 Action 而不是 Final Answer
        for _ in range(10):
            mock_llm.add_response(
                'Thought: Still thinking...\nAction: search\nAction Input: {"query": "x"}'
            )
        result = await agent.run(
            query="Something complex",
            context={"tool_names": ["search"]},
        )
        assert result.success is False
        assert "最大步数" in (result.error or "")

    async def test_llm_failure_handling(self, mock_tools: MockToolInvoker) -> None:
        """LLM 调用失败应返回错误。"""
        bad_llm = MockLLM()

        async def raise_error(messages, **kwargs) -> str:
            raise RuntimeError("LLM connection failed")

        bad_llm.acomplete = raise_error  # type: ignore[assignment]

        agent = ReActAgent(llm=bad_llm, tools=mock_tools, memory=None, max_steps=5)
        result = await agent.run("test", context={"tool_names": []})
        assert result.success is False
        assert "LLM 调用失败" in (result.error or "")

    async def test_memory_integration(
        self,
        mock_llm: MockLLM,
        mock_tools: MockToolInvoker,
    ) -> None:
        """记忆模块集成测试。"""
        memory = MockMemoryLike()
        memory.set_snippets(["User previously asked about ML", "User likes short answers"])
        mock_llm.add_response(
            "Thought: Done.\nFinal Answer: Here is the answer."
        )

        agent = ReActAgent(
            llm=mock_llm,
            tools=mock_tools,
            memory=memory,
            max_steps=5,
            session_id="test-session",
        )
        result = await agent.run(
            query="What's new in AI?",
            context={"tool_names": [], "session_id": "test-session"},
        )
        assert result.success is True
        # 应存储了对话轮次
        assert "test-session" in memory.store
        stored = memory.store["test-session"]
        assert any("Here is the answer" in str(r) for r in stored)

    async def test_memory_failure_tolerant(
        self,
        mock_llm: MockLLM,
        mock_tools: MockToolInvoker,
    ) -> None:
        """记忆失败不应阻塞 Agent。"""
        memory = MockMemoryLike()

        async def fail_memory(*args, **kwargs):
            raise RuntimeError("Memory unavailable")

        memory.get_relevant = fail_memory  # type: ignore[assignment]
        mock_llm.add_response(
            "Thought: I'll answer anyway.\nFinal Answer: Answer without memory."
        )

        agent = ReActAgent(
            llm=mock_llm, tools=mock_tools, memory=memory, max_steps=5
        )
        result = await agent.run("test", context={"tool_names": []})
        assert result.success is True
        assert "Answer without memory" in result.final_answer

    async def test_tool_not_in_allowed_list(
        self, agent: ReActAgent, mock_llm: MockLLM
    ) -> None:
        """调用不在允许列表中的工具应返回观察错误。"""
        mock_llm.add_response(
            'Thought: I need database.\nAction: database\nAction Input: {"sql": "SELECT 1"}'
        )
        result = await agent.run(
            query="Query data",
            context={"tool_names": ["search", "calculator"]},  # database not in list
        )
        assert result.success is False
        obs = result.steps[0].get("observation", "")
        assert "不在允许列表" in obs

    async def test_trace_callback_invocation(
        self, agent: ReActAgent, mock_llm: MockLLM
    ) -> None:
        """追踪回调应在每步被调用。"""
        trace_events: list[dict] = []

        async def trace_cb(rec):
            trace_events.append(rec)

        mock_llm.add_response(
            "Thought: Done.\nFinal Answer: Tracked answer."
        )
        result = await agent.run(
            query="test",
            context={"tool_names": [], "trace_callback": trace_cb},
        )
        assert result.success is True
        assert len(trace_events) >= 1

    async def test_custom_session_id_from_context(
        self, agent: ReActAgent, mock_llm: MockLLM
    ) -> None:
        """context 中的 session_id 覆盖实例默认值。"""
        mock_llm.add_response(
            "Thought: Done.\nFinal Answer: Answer"
        )
        result = await agent.run(
            query="test",
            context={"tool_names": [], "session_id": "custom-session"},
        )
        assert result.success is True
