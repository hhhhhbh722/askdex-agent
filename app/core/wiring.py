# -*- coding: utf-8 -*-
"""应用接线：Embedding + Milvus + Redis + Agent。"""
from __future__ import annotations

import httpx
from loguru import logger

_state: dict = {}

def get_state(): return _state


# ---- Embedding ----

class EmbeddingAPI:
    def __init__(self, base_url: str, api_key: str, model: str, dim: int = 1024):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim

    def _headers(self): return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @staticmethod
    def _raise_for_embedding_error(response) -> None:
        try:
            response.raise_for_status()
        except Exception as exc:
            body = getattr(response, "text", "") or ""
            raise RuntimeError(f"Embedding API 请求失败: HTTP {response.status_code} {body[:500]}") from exc

    def encode(self, texts: list[str]) -> list[list[float]]:
        import requests
        r = requests.post(f"{self.base_url}/embeddings", json={"model": self.model, "input": texts},
                          headers=self._headers(), timeout=60)
        self._raise_for_embedding_error(r)
        d = r.json()
        return [x["embedding"] for x in sorted(d["data"], key=lambda x: x["index"])]

    async def aencode(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient() as c:
            r = await c.post(f"{self.base_url}/embeddings", json={"model": self.model, "input": texts},
                             headers=self._headers(), timeout=60)
            self._raise_for_embedding_error(r)
            d = r.json()
            return [x["embedding"] for x in sorted(d["data"], key=lambda x: x["index"])]

    def embed_query(self, text: str) -> list[float]:
        return self.encode([text])[0]


# ---- Milvus ----

async def init_milvus(settings) -> any:
    from app.infrastructure.vectordb.milvus_client import MilvusManager
    try:
        m = MilvusManager(host=settings.milvus_host, port=str(settings.milvus_port))
        await m.create_collection(settings.milvus_collection_name, dim=settings.embedding_dim)
        logger.info("✅ Milvus: {}:{} dim={}", settings.milvus_host, settings.milvus_port, settings.embedding_dim)
        return m
    except Exception as e:
        logger.warning("⚠️ Milvus 不可用: {}", e)
        return None


# ---- Redis ----

def init_redis(settings) -> any:
    try:
        import redis.asyncio as aioredis
        c = aioredis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=3)
        logger.info("✅ Redis: {}", settings.redis_url)
        return c
    except Exception as e:
        logger.warning("⚠️ Redis 不可用: {}", e)
        return None


# ---- RAG 检索 ----

async def rag_search(query: str, top_k: int = 5, filters: dict | None = None) -> list[dict]:
    """GraphRAG 检索：KG 事实召回 + Milvus/BM25 混合检索 + RRF 融合。"""
    emb = _state.get("embedding")
    milvus = _state.get("milvus")
    if not emb or not milvus: return []
    s = _state["settings"]
    from app.core.retrieval.service import rag_search as retrieval_search
    llm = _state.get("agent_llm")
    reranker = _state.get("reranker")
    return await retrieval_search(
        query,
        embedding=emb,
        milvus=milvus,
        settings=s,
        llm=llm,
        reranker=reranker,
        top_k=top_k,
        filters=filters,
    )


# ---- ETL → Milvus ----

async def embed_chunks(
    chunks: list[str],
    chunk_ids: list[str],
    source: str = "",
    document_id: str = "",
    entity_name: str = "",
    normalized_entity_name: str = "",
    group: str = "",
    parent_group: str = "",
    child_group: str = "",
) -> int:
    emb, milvus = _state.get("embedding"), _state.get("milvus")
    if not emb or not milvus or not chunks: return 0
    s = _state["settings"]
    vecs = await emb.aencode(chunks)
    meta = [{
        "id": cid,
        "content": text[:65000],
        "source": source,
        "document_id": document_id,
        "entity_name": entity_name,
        "normalized_entity_name": normalized_entity_name,
        "group": group,
        "parent_group": parent_group,
        "child_group": child_group,
        "chunk_index": i,
    }
            for i, (cid, text) in enumerate(zip(chunk_ids, chunks))]
    await milvus.insert(s.milvus_collection_name, vecs, meta)
    return len(vecs)


# ---- 启动 ----

async def wire_app(app) -> None:
    from app.config import get_settings
    s = get_settings()

    # Embedding
    api_key = s.embedding_api_key or s.openai_api_key
    emb = EmbeddingAPI(s.embedding_api_base or s.openai_api_base, api_key, s.embedding_model, s.embedding_dim)
    logger.info("✅ Embedding: {} dim={}", s.embedding_model, s.embedding_dim)

    milvus = await init_milvus(s)
    redis = init_redis(s)

    app.state.embedding = emb
    app.state.milvus = milvus
    app.state.redis = redis

    # Reranker（DashScope API）
    reranker = None
    if s.embedding_api_key:
        from app.core.retrieval.reranker import Reranker
        reranker = Reranker(api_key=s.embedding_api_key, model="qwen3-vl-rerank")
        logger.info("✅ Reranker: qwen3-vl-rerank")

    _state.update(embedding=emb, milvus=milvus, redis=redis, settings=s,
                  reranker=reranker)

    # Agent（内部会设置 agent_llm）
    agent = build_agent(s, emb, milvus, redis)
    app.state.agent = agent
    _state["agent"] = agent

    logger.info("✅ 组件就绪: emb=api milvus={} redis={} reranker={} agent={}",
                "ok" if milvus else "no", "ok" if redis else "no",
                "qwen3-vl-rerank" if reranker else "no", type(agent).__name__)


# ---- Agent 构造 ----

def build_agent(settings, embedding, milvus, redis):
    """构造 AgentOrchestrator，注入所有工具。"""
    from app.core.agent.orchestrator import AgentOrchestrator, InMemoryTracer
    from app.core.tools.registry import ToolRegistry
    from app.core.tools.builtin.calculator import CalculatorTool
    from app.core.tools.builtin.knowledge_base import KnowledgeBaseTool
    from app.core.tools.builtin.search import WebSearchTool
    from app.infrastructure.llm.model_router import ModelConfig, ModelRouter

    # LLM
    router = ModelRouter([ModelConfig(
        model_id=settings.openai_model, api_key=settings.openai_api_key,
        base_url=settings.openai_api_base, priority=0, weight=1.0,
    )])
    # 保持引用给 HyDE 用
    adapter = _LLMAdapter(router)
    _state["agent_llm"] = adapter

    # 工具
    tools = ToolRegistry(timeout_seconds=90.0)
    tools.register(CalculatorTool())
    tools.register(WebSearchTool())
    tools.register(KnowledgeBaseTool(rag_search))  # 知识库检索工具

    # 记忆
    from app.core.memory.short_term import ShortTermMemory
    from app.core.memory.manager import MemoryManager
    stm = ShortTermMemory(
        redis_client=redis,
        llm=None,
        window_size=20,
        max_tokens=4000,
        ttl_seconds=settings.redis_memory_ttl_seconds,
    )
    ltm = _SimpleLTM()  # 长期记忆简化版
    memory = MemoryManager(short_term=stm, long_term=ltm)

    # 配置
    config = _AgentConfig()
    tracer = _build_agent_tracer(redis, settings, InMemoryTracer)
    _state["agent_tracer"] = tracer

    return AgentOrchestrator(
        config=config, model_router=_ModelRouterAdapter(router),
        memory_manager=memory, tool_registry=tools, tracer=tracer,
    )


# ---- 适配器 ----

def _build_agent_tracer(redis, settings, fallback_cls):
    if redis:
        try:
            from app.infrastructure.trace.redis_trace_store import RedisAgentTracer

            return RedisAgentTracer(
                redis_client=redis,
                ttl_seconds=settings.redis_memory_ttl_seconds,
            )
        except Exception as e:
            logger.warning("Redis Agent trace store unavailable, fallback to memory: {}", e)
    return fallback_cls()


class _ModelRouterAdapter:
    """将 ModelRouter 适配为 Orchestrator 期望的 get_llm(purpose) 接口。"""
    def __init__(self, router):
        self._router = router
        self._llm = _LLMAdapter(router)

    def get_llm(self, purpose: str):
        return self._llm


class _LLMAdapter:
    """将 ModelRouter.chat() 适配为 acomplete(messages) 接口。"""
    def __init__(self, router):
        self._r = router

    async def acomplete(self, messages, **kw):
        msgs = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]
        resp = await self._r.chat(msgs, temperature=kw.get("temperature", 0.2))
        return resp.content


class _AgentConfig:
    def get(self, key, default=None):
        return {
            "enable_reflection": True, "react_max_steps": 8,
            "max_replan_attempts": 2, "fallback_react_on_plan_failure": True,
            "reflection_min_quality": 60,
        }.get(key, default)


class _SimpleLTM:
    """简化长期记忆，无持久化。"""
    async def recall(self, query, session_id, top_k=5):
        return []
    async def store(self, session_id, content, metadata=None):
        return ""
    async def forget(self, memory_id):
        pass
