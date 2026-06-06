# -*- coding: utf-8 -*-
"""对话 API：Agent 编排（非流式）+ RAG 直接（流式）。"""
from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from openai import AsyncOpenAI

from app.config import get_settings
from app.core.wiring import get_state, rag_search
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])

# 内存统计
_stats = {"total_requests": 0, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "traces": []}


async def _rag_context(query: str, top_k: int = 5) -> str:
    results = await rag_search(query, top_k)
    if not results: return ""
    lines = ["\n## 知识库相关内容\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.get('content', '')[:500]}")
    return "\n\n".join(lines)


@router.get("/metrics")
async def metrics() -> dict:
    """返回实时统计（内存累计）。"""
    return _stats


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """非流式：Agent 编排（ReAct 推理 + 工具调用 + 知识库检索）。"""
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(503, "未配置 API Key")

    user_text = request.messages[-1].content if request.messages else ""
    started = time.perf_counter()

    # 直接调 LLM（Agent 在后台处理工具调用）
    agent = get_state().get("agent")
    if agent:
        try:
            from app.core.agent.orchestrator import AgentResponse, IntentContext
            preferred_mode = request.mode or "react"
            resp = await agent.run(
                user_text,
                session_id=request.conversation_id or str(uuid.uuid4()),
                mode=preferred_mode,
                intent=IntentContext(
                    intent="general",
                    confidence=1.0,
                    slots={},
                    preferred_mode=preferred_mode,
                    allowed_tools=None,
                ),
            )
            if isinstance(resp, AgentResponse):
                _record_agent_trace(
                    trace_id=resp.trace_id,
                    operation=f"agent.{resp.mode_used}",
                    started=started,
                    error=resp.error,
                    steps=resp.steps,
                )
                return ChatResponse(id=str(uuid.uuid4()), model=settings.openai_model,
                                    content=resp.answer, trace_id=resp.trace_id,
                                    steps=resp.steps, mode=resp.mode_used)
            answer = str(resp)
            return ChatResponse(id=str(uuid.uuid4()), model=settings.openai_model,
                                content=answer, steps=[], mode="")
        except Exception as e:
            logger.warning("Agent 调用失败，降级直连: {}", e)

    # 降级：直接 LLM + RAG
    rag = await _rag_context(user_text)
    messages = [m.model_dump() for m in request.messages]
    if rag: messages.insert(-1, {"role": "system", "content": rag})

    client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_api_base or None)
    resp = await client.chat.completions.create(
        model=settings.openai_model, messages=messages,
        temperature=request.temperature, max_tokens=request.max_tokens)
    _record_stats(resp.usage)
    return ChatResponse(id=str(uuid.uuid4()), model=settings.openai_model,
                        content=resp.choices[0].message.content or "",
                        usage={"prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                               "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                               "total_tokens": resp.usage.total_tokens if resp.usage else 0})


def _record_agent_trace(
    trace_id: str,
    operation: str,
    started: float,
    error: str | None,
    steps: list[dict],
) -> None:
    _stats["total_requests"] += 1
    duration_ms = int((time.perf_counter() - started) * 1000)
    _stats["traces"].insert(0, {
        "id": trace_id,
        "operation": operation,
        "duration": duration_ms,
        "error": error,
        "spans": [
            {
                "id": str(step.get("step", step.get("subtask_id", idx))),
                "operation": str(step.get("phase") or step.get("action") or step.get("subtask_id") or "step"),
                "duration": int(step.get("tool_duration_ms") or 0),
            }
            for idx, step in enumerate(steps[:20])
        ],
    })
    _stats["traces"] = _stats["traces"][:50]


@router.get("/chat/memory")
async def chat_memory(session_id: str = Query(...)) -> dict:
    """Debug endpoint for the current Redis short-term conversation memory."""
    redis_client = get_state().get("redis")
    if not redis_client:
        raise HTTPException(503, "Redis 未启用")

    key = f"stm:session:{session_id}"
    raw_items = await redis_client.lrange(key, 0, -1)
    items = []
    for raw in raw_items:
        try:
            items.append(json.loads(raw))
        except Exception:
            items.append({"raw": str(raw)})
    return {"session_id": session_id, "key": key, "count": len(items), "messages": items}


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """SSE 流式：直接 LLM + RAG（Agent 暂不支持流式）。"""
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(503, "未配置 API Key")

    user_text = request.messages[-1].content if request.messages else ""
    rag = await _rag_context(user_text)

    async def gen() -> AsyncIterator[bytes]:
        client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_api_base or None)
        messages = [m.model_dump() for m in request.messages]
        if rag: messages.insert(-1, {"role": "system", "content": rag})
        try:
            stream = await client.chat.completions.create(
                model=settings.openai_model, messages=messages,
                temperature=request.temperature, max_tokens=request.max_tokens, stream=True)
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield b"data: " + json.dumps({"content": delta}, ensure_ascii=False).encode() + b"\n\n"
            yield b"data: " + json.dumps({"done": True}, ensure_ascii=False).encode() + b"\n\n"
        except Exception as e:
            yield b"data: " + json.dumps({"error": str(e)}, ensure_ascii=False).encode() + b"\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


def _record_stats(usage) -> None:
    if not usage: return
    _stats["total_requests"] += 1
    _stats["total_tokens"] += usage.total_tokens or 0
    _stats["prompt_tokens"] += usage.prompt_tokens or 0
    _stats["completion_tokens"] += usage.completion_tokens or 0
