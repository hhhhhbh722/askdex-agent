# -*- coding: utf-8 -*-
"""AgentOrchestrator 编排器测试。"""

from __future__ import annotations

import pytest

from app.core.agent.orchestrator import (
    AgentOrchestrator,
    AgentResponse,
    InMemoryTracer,
    IntentContext,
    _LLMAdapter,
)
from tests.conftest import (
    MockLLM,
    MockMemoryLike,
    MockModelRouterForOrchestrator,
    MockOrchestratorConfig,
    MockTracerForAgent,
)


# ============================================================================
# 夹具
# ============================================================================


class MockToolRegistryForOrch:
    """实现 ToolRegistry Protocol（list_tool_names + invoke）。"""

    def __init__(self, tool_names: list[str] | None = None) -> None:
        self._names = tool_names or ["search", "calculator"]

    def list_tool_names(self) -> list[str]:
        return list(self._names)

    async def invoke(self, name: str, arguments: dict) -> str:
        return f"Result from {name}: {arguments}"


class MockMemoryManagerForOrch:
    """实现 MemoryManager Protocol。"""

    async def get_relevant(self, session_id: str, query: str, limit: int = 8) -> list[str]:
        return []

    async def append_turn(self, session_id: str, role: str, content: str, metadata=None) -> None:
        pass


@pytest.fixture
def config() -> MockOrchestratorConfig:
    return MockOrchestratorConfig({
        "enable_reflection": True,
        "react_max_steps": 3,
        "max_replan_attempts": 1,
        "fallback_react_on_plan_failure": True,
        "reflection_min_quality": 60,
    })


@pytest.fixture
def model_router() -> MockModelRouterForOrchestrator:
    r = MockModelRouterForOrchestrator()
    r.set_llm("react", MockLLM())
    r.set_llm("planner", MockLLM())
    r.set_llm("reflection", MockLLM())
    return r


@pytest.fixture
def memory_manager() -> MockMemoryManagerForOrch:
    return MockMemoryManagerForOrch()


@pytest.fixture
def tool_registry() -> MockToolRegistryForOrch:
    return MockToolRegistryForOrch()


@pytest.fixture
def tracer() -> MockTracerForAgent:
    return MockTracerForAgent()


@pytest.fixture
def orchestrator(
    config: MockOrchestratorConfig,
    model_router: MockModelRouterForOrchestrator,
    memory_manager: MockMemoryManagerForOrch,
    tool_registry: MockToolRegistryForOrch,
    tracer: MockTracerForAgent,
) -> AgentOrchestrator:
    return AgentOrchestrator(
        config=config,
        model_router=model_router,
        memory_manager=memory_manager,
        tool_registry=tool_registry,
        tracer=tracer,
    )


# ============================================================================
# 测试
# ============================================================================


class TestOrchestratorReactMode:
    """ReAct 模式测试。"""

    async def test_react_mode_success(
        self, orchestrator: AgentOrchestrator, model_router: MockModelRouterForOrchestrator
    ) -> None:
        react_llm = MockLLM()
        react_llm.add_response(
            "Thought: I can answer.\nFinal Answer: AI Agent is an autonomous system."
        )
        model_router.set_llm("react", react_llm)
        # reflection LLM
        reflect_llm = MockLLM()
        reflect_llm.add_response(
            '{"quality_score":80,"is_complete":true,"likely_hallucination":false,'
            '"hallucination_reasons":[],"completeness_notes":"OK",'
            '"suggestions":[],"summary":"Good"}'
        )
        model_router.set_llm("reflection", reflect_llm)

        resp = await orchestrator.run("What is AI?", "sess-1", mode="react")
        assert isinstance(resp, AgentResponse)
        assert resp.mode_used == "react"
        assert "AI Agent" in resp.answer


class TestOrchestratorPlanExecuteMode:
    """Plan-and-Execute 模式测试。"""

    async def test_plan_execute_success(
        self, orchestrator: AgentOrchestrator, model_router: MockModelRouterForOrchestrator
    ) -> None:
        import json
        # Planner LLM: 生成计划
        planner_llm = MockLLM()
        planner_llm.add_response(json.dumps({
            "subtasks": [{
                "id": "t1", "title": "Think",
                "description": "Reason about the question",
                "action_type": "reasoning", "tool_name": None, "tool_args_hint": None
            }]
        }))
        planner_llm.add_response("Executing reasoning step.")  # execute reasoning
        planner_llm.add_response("Final answer from plan-execute.")  # summary
        model_router.set_llm("planner", planner_llm)
        # Reflection LLM
        reflect_llm = MockLLM()
        reflect_llm.add_response(
            '{"quality_score":80,"is_complete":true,"likely_hallucination":false,'
            '"hallucination_reasons":[],"completeness_notes":"OK",'
            '"suggestions":[],"summary":"Good"}'
        )
        model_router.set_llm("reflection", reflect_llm)

        resp = await orchestrator.run("What is AI?", "sess-1", mode="plan_execute")
        assert resp.mode_used == "plan_execute"
        assert resp.success is True


class TestOrchestratorIntent:
    """意图上下文的传递和影响测试。"""

    async def test_intent_preferred_mode_overrides(
        self, orchestrator: AgentOrchestrator, model_router: MockModelRouterForOrchestrator
    ) -> None:
        """IntentContext.preferred_mode 覆盖 run() 的 mode 参数。"""
        react_llm = MockLLM()
        react_llm.add_response(
            "Thought: Done.\nFinal Answer: Answer via react fallback."
        )
        model_router.set_llm("react", react_llm)

        intent = IntentContext(
            intent="task",
            preferred_mode="react",
            allowed_tools=["search"],
        )
        resp = await orchestrator.run(
            "query", "sess-1", mode="plan_execute", intent=intent
        )
        # intent 强制 react 模式
        assert resp.mode_used == "react"

    async def test_intent_allowed_tools_filtered(
        self, orchestrator: AgentOrchestrator, model_router: MockModelRouterForOrchestrator
    ) -> None:
        """IntentContext.allowed_tools 限制工具列表。"""
        react_llm = MockLLM()
        react_llm.add_response(
            'Thought: Using search.\nAction: search\nAction Input: {"query": "AI"}'
        )
        react_llm.add_response(
            "Thought: Got results.\nFinal Answer: Search complete."
        )
        model_router.set_llm("react", react_llm)

        intent = IntentContext(
            intent="task",
            allowed_tools=["search"],  # only search allowed
        )
        resp = await orchestrator.run("query", "sess-1", intent=intent)
        assert resp.success is True


class TestOrchestratorFallback:
    """降级机制测试。"""

    async def test_fallback_when_plan_fails(
        self, orchestrator: AgentOrchestrator, model_router: MockModelRouterForOrchestrator
    ) -> None:
        """plan 失败时回退到 react。"""
        # planner LLM 失败 → fallback plan
        planner_llm = MockLLM()

        async def fail_planner(messages, **kwargs):
            raise RuntimeError("Planner unavailable")

        planner_llm.acomplete = fail_planner  # type: ignore[assignment]
        model_router.set_llm("planner", planner_llm)

        # react LLM 需要能够回退成功
        react_llm = MockLLM()
        react_llm.add_response(
            "Thought: Fallback mode.\nFinal Answer: Answered via react fallback."
        )
        model_router.set_llm("react", react_llm)

        # reflection
        reflect_llm = MockLLM()
        reflect_llm.add_response(
            '{"quality_score":70,"is_complete":true,"likely_hallucination":false,'
            '"hallucination_reasons":[],"completeness_notes":"OK",'
            '"suggestions":[],"summary":"OK"}'
        )
        model_router.set_llm("reflection", reflect_llm)

        resp = await orchestrator.run("query", "sess-1", mode="plan_execute")
        assert resp.degraded is True
        assert "react fallback" in resp.answer


class TestOrchestratorEdgeCases:
    """编排器边界情况测试。"""

    async def test_top_level_exception_returns_error_response(
        self, orchestrator: AgentOrchestrator, model_router: MockModelRouterForOrchestrator
    ) -> None:
        """顶层异常不应崩溃，返回错误响应。"""
        # react LLM 也抛异常
        react_llm = MockLLM()

        async def fail_all(messages, **kwargs):
            raise RuntimeError("Everything fails")

        react_llm.acomplete = fail_all  # type: ignore[assignment]
        model_router.set_llm("react", react_llm)

        resp = await orchestrator.run("query", "sess-1")
        assert isinstance(resp, AgentResponse)
        assert resp.success is False

    async def test_in_memory_tracer(self) -> None:
        """InMemoryTracer 能正常工作。"""
        t = InMemoryTracer()
        trace_id = t.new_trace_id()
        span = t.start_span("test", trace_id)
        t.end_span(span)
        assert len(t.events) >= 1


class TestLLMAdapter:
    """_LLMAdapter 适配器测试。"""

    async def test_adapter_with_acomplete(self) -> None:
        class HasAcomplete:
            async def acomplete(self, messages, **kwargs) -> str:
                return "result"

        adapter = _LLMAdapter(HasAcomplete())
        result = await adapter.acomplete([{"role": "user", "content": "hi"}])
        assert result == "result"

    async def test_adapter_without_methods_raises(self) -> None:
        class NoMethods:
            pass

        adapter = _LLMAdapter(NoMethods())
        with pytest.raises(TypeError):
            await adapter.acomplete([{"role": "user", "content": "hi"}])
