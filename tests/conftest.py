# -*- coding: utf-8 -*-
"""共享测试夹具与 Protocol 模拟类。"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.enums import MessageRole
from app.models.schemas import (
    ChatMessage,
    MemoryItem,
    Message,
    RAGResponse,
    RetrievalResult,
)


# ============================================================================
# 工厂辅助函数
# ============================================================================


def make_retrieval_result(
    id: str = "doc-1",
    content: str = "Test document content",
    score: float = 0.85,
    source: str = "vector",
    metadata: dict | None = None,
) -> RetrievalResult:
    """创建供测试用的 RetrievalResult。"""
    return RetrievalResult(
        id=id,
        content=content,
        score=score,
        metadata=metadata or {"rank": 0},
        source=source,
    )


def make_message(
    role: MessageRole = MessageRole.USER,
    content: str = "Test message",
    metadata: dict | None = None,
) -> Message:
    """创建供测试用的 Message。"""
    return Message(role=role, content=content, metadata=metadata or {})


def make_memory_item(
    id: str = "mem-1",
    content: str = "Memory snippet",
    score: float = 0.9,
    metadata: dict | None = None,
) -> MemoryItem:
    """创建供测试用的 MemoryItem。"""
    return MemoryItem(id=id, content=content, score=score, metadata=metadata or {})


# ============================================================================
# Protocol 模拟类 — 用于 Agent 测试
# ============================================================================


class MockLLM:
    """实现 LLMCallable Protocol：async acomplete(messages, **kw) -> str。

    使用方式：
        llm = MockLLM()
        llm.add_response("搜索", "Thought: Done\\nFinal Answer: Found it")
        result = await llm.acomplete([...])

    每次调用 acomplete 会按索引使用 responses；若耗尽则返回默认回复。
    """

    def __init__(self, default_response: str = "Thought: I have enough information.\nFinal Answer: Test answer.") -> None:
        self.responses: list[str] = []
        self.default_response = default_response
        self.calls: list[dict[str, Any]] = []
        self._idx = 0

    def add_response(self, response: str) -> None:
        self.responses.append(response)

    def add_conditional_response(self, contains: str, response: str) -> None:
        """仅当 messages 中包含特定子串时返回指定响应。"""
        self.responses.append(f"__cond__{contains}__|__{response}")

    async def acomplete(self, messages: Sequence[Dict[str, str]], **kwargs: Any) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self._idx < len(self.responses):
            resp = self.responses[self._idx]
            self._idx += 1
            if resp.startswith("__cond__"):
                parts = resp.split("__|__", 1)
                cond = parts[0].replace("__cond__", "")
                text = parts[1] if len(parts) > 1 else self.default_response
                # 检查条件
                all_text = " ".join(m.get("content", "") for m in messages)
                if cond in all_text:
                    return text
                # 条件不匹配，继续使用默认
            return resp
        return self.default_response


class MockToolInvoker:
    """实现 ToolInvoker Protocol：async invoke(name, arguments) -> str。"""

    def __init__(self, results: dict[str, str] | None = None) -> None:
        self.results = results or {}
        self.calls: list[dict[str, Any]] = []
        self.default_result = '{"result": "ok"}'

    async def invoke(self, name: str, arguments: Dict[str, Any]) -> str:
        self.calls.append({"name": name, "arguments": arguments})
        return self.results.get(name, self.default_result)


class MockMemoryLike:
    """实现 MemoryLike Protocol。"""

    def __init__(self) -> None:
        self._store: dict[str, list[dict[str, Any]]] = {}
        self._snippets: list[str] = []

    def set_snippets(self, snippets: list[str]) -> None:
        self._snippets = list(snippets)

    async def get_relevant(self, session_id: str, query: str, limit: int = 8) -> list[str]:
        return self._snippets[:limit]

    async def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append(
            {"role": role, "content": content, "metadata": metadata or {}}
        )

    @property
    def store(self) -> dict[str, list[dict[str, Any]]]:
        return self._store


class MockOrchestratorConfig:
    """实现 OrchestratorConfig Protocol。"""

    def __init__(self, overrides: dict[str, Any] | None = None) -> None:
        self._values = {
            "enable_reflection": True,
            "react_max_steps": 5,
            "max_replan_attempts": 2,
            "fallback_react_on_plan_failure": True,
            "reflection_min_quality": 60,
            **(overrides or {}),
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)


class MockTracerForAgent:
    """实现 Tracer Protocol（用于编排器测试）。事件存储在 self.events。"""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._spans: dict[str, dict[str, Any]] = {}

    def new_trace_id(self) -> str:
        return str(uuid.uuid4())

    def start_span(
        self,
        name: str,
        trace_id: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> dict[str, Any]:
        span = {"name": name, "trace_id": trace_id, "attributes": attributes or {}}
        self._spans[trace_id] = span
        return span

    def end_span(self, span: Any, error: Optional[BaseException] = None) -> None:
        self.events.append(
            {"type": "end_span", "span": span, "error": str(error) if error else None}
        )

    def log_event(self, trace_id: str, name: str, payload: Dict[str, Any]) -> None:
        self.events.append(
            {"type": "event", "trace_id": trace_id, "name": name, "payload": payload}
        )


class MockModelRouterForOrchestrator:
    """实现编排器用的 ModelRouter Protocol。"""

    def __init__(self, llm_map: dict[str, MockLLM] | None = None) -> None:
        self._llm_map = llm_map or {}

    def set_llm(self, purpose: str, llm: MockLLM) -> None:
        self._llm_map[purpose] = llm

    def get_llm(self, purpose: str) -> Any:
        return self._llm_map.get(purpose, MockLLM())


# ============================================================================
# RAG 相关模拟
# ============================================================================


class MockEmbedding:
    """实现 EmbeddingProtocol。返回固定维度向量。"""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self.calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        # 返回基于文本哈希的确定性向量
        import hashlib
        h = hashlib.md5(text.encode())
        seed = int(h.hexdigest()[:8], 16)
        return [(seed % 1000) / 1000.0 for _ in range(self.dim)]


class MockMilvusCollection:
    """实现 LTMCollectionProtocol 的内存 Milvus。"""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._id_counter = 0

    def insert(self, data: list[dict[str, Any]], **kwargs: Any) -> Any:
        for row in data:
            if isinstance(row, dict):
                self.rows.append(dict(row))
            elif isinstance(row, list):
                self.rows.append({"pk": self._id_counter, "data": row})
                self._id_counter += 1
        return type("InsertResult", (), {"insert_count": len(data)})()

    def search(
        self,
        data: list[list[float]],
        anns_field: str,
        param: dict[str, Any],
        limit: int,
        expr: str | None = None,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        # 返回模拟命中列表
        results = []
        for row in self.rows[:limit]:
            hit = type(
                "Hit",
                (),
                {
                    "id": row.get("pk", "unknown"),
                    "distance": 0.5,
                    "entity": {
                        "pk": row.get("pk", "unknown"),
                        "content": row.get("content", ""),
                        "meta": row.get("meta", "{}"),
                    },
                },
            )()
            results.append(hit)
        return [results]

    def delete(self, expr: str, **kwargs: Any) -> Any:
        self.rows = [r for r in self.rows if str(r.get("pk", "")) not in expr]
        return type("DeleteResult", (), {"delete_count": 1})()

    def flush(self, **kwargs: Any) -> Any:
        return None


class MockRAGLLM:
    """实现 RAGLLMProtocol：async ainvoke(input, **kw) -> 含 .content 的对象。"""

    def __init__(self, content: str = "Test answer [1]") -> None:
        self.content = content
        self.calls: list[Any] = []

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        self.calls.append({"input": input, "kwargs": kwargs})
        return type("LLMOutput", (), {"content": self.content})()


# ============================================================================
# 记忆模块模拟
# ============================================================================


class MockRedisClient:
    """模拟 redis.asyncio.Redis 的基础列表操作。"""

    def __init__(self) -> None:
        self._lists: dict[str, list[bytes]] = {}
        self._strings: dict[str, str] = {}

    async def lrange(self, key: str, start: int, end: int) -> list[bytes]:
        lst = self._lists.get(key, [])
        if end == -1:
            return lst[start:]
        return lst[start:end + 1]

    async def rpush(self, key: str, value: str) -> int:
        if key not in self._lists:
            self._lists[key] = []
        self._lists[key].append(value.encode("utf-8"))
        return len(self._lists[key])

    async def llen(self, key: str) -> int:
        return len(self._lists.get(key, []))

    async def delete(self, key: str) -> int:
        if key in self._lists:
            del self._lists[key]
            return 1
        return 0

    async def get(self, key: str) -> str | None:
        return self._strings.get(key)

    async def set(self, key: str, value: str) -> bool:
        self._strings[key] = value
        return True

    async def setex(self, key: str, time: int, value: str) -> bool:
        self._strings[key] = value
        return True


# ============================================================================
# Pytest 夹具
# ============================================================================


@pytest.fixture
def settings_override(monkeypatch):
    """设置测试环境变量并清除 settings 缓存。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-12345")
    monkeypatch.setenv("APP_ENV", "testing")
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def mock_llm() -> MockLLM:
    """返回可配置的 LLM 模拟。"""
    return MockLLM()


@pytest.fixture
def mock_tools() -> MockToolInvoker:
    """返回可配置的工具模拟。"""
    return MockToolInvoker()


@pytest.fixture
def mock_memory() -> MockMemoryLike:
    """返回可配置的记忆模拟。"""
    return MockMemoryLike()


@pytest.fixture
def mock_config() -> MockOrchestratorConfig:
    """返回带合理默认值的编排器配置模拟。"""
    return MockOrchestratorConfig()


@pytest.fixture
def mock_tracer() -> MockTracerForAgent:
    """返回内存事件存储的追踪器模拟。"""
    return MockTracerForAgent()


@pytest.fixture
def mock_model_router() -> MockModelRouterForOrchestrator:
    """返回编排器用的模型路由器模拟。"""
    r = MockModelRouterForOrchestrator()
    r.set_llm("react", MockLLM())
    r.set_llm("planner", MockLLM())
    r.set_llm("reflection", MockLLM())
    return r


@pytest.fixture
def sample_retrieval_results() -> list[RetrievalResult]:
    """返回用于 RAG 测试的标准检索结果列表。"""
    return [
        make_retrieval_result(id="doc-1", content="Document one about AI agents", score=0.95),
        make_retrieval_result(id="doc-2", content="Document two about RAG pipelines", score=0.80),
        make_retrieval_result(id="doc-3", content="Document three about vector search", score=0.65),
    ]


@pytest.fixture
def sample_chat_messages() -> list[ChatMessage]:
    """返回用于 API 测试的标准聊天消息。"""
    return [ChatMessage(role="user", content="What is an AI agent?")]


@pytest.fixture
def sample_chat_messages_long() -> list[ChatMessage]:
    """返回多轮对话消息。"""
    return [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Hello!"),
        ChatMessage(role="assistant", content="Hi there! How can I help?"),
        ChatMessage(role="user", content="What is an AI agent?"),
    ]


@pytest.fixture
def mock_embedding() -> MockEmbedding:
    """返回确定性嵌入模型。"""
    return MockEmbedding()


@pytest.fixture
def mock_milvus_collection() -> MockMilvusCollection:
    """返回内存 Milvus 集合。"""
    return MockMilvusCollection()


@pytest.fixture
def mock_redis_client() -> MockRedisClient:
    """返回内存 Redis 客户端。"""
    return MockRedisClient()


@pytest.fixture
def mock_rag_llm() -> MockRAGLLM:
    """返回 RAG 使用的 mock LLM。"""
    return MockRAGLLM()
