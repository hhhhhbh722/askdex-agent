# -*- coding: utf-8 -*-
"""Redis-backed Agent trace store.

The orchestrator tracer interface is synchronous, while the Redis client is
async. This adapter schedules small background writes so Agent execution is not
blocked by trace persistence.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from loguru import logger

from app.infrastructure.observability import current_otel_span_id, current_otel_trace_id


class RedisAgentTracer:
    """Persist Agent run state and step replay records in Redis."""

    def __init__(
        self,
        redis_client: Any,
        ttl_seconds: int = 86400,
        max_events: int = 200,
        max_steps: int = 100,
    ) -> None:
        self._redis = redis_client
        self._ttl = max(int(ttl_seconds or 86400), 60)
        self._max_events = max_events
        self._max_steps = max_steps

    def new_trace_id(self) -> str:
        return current_otel_trace_id() or str(uuid.uuid4())

    def start_span(
        self,
        name: str,
        trace_id: str,
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        span = {
            "name": name,
            "trace_id": trace_id,
            "span_id": current_otel_span_id() or uuid.uuid4().hex[:16],
            "attributes": attributes or {},
            "started_at": time.time(),
            "_perf_start": time.perf_counter(),
        }
        self._schedule(self._start_trace(span))
        return span

    def end_span(self, span: Any, error: BaseException | None = None) -> None:
        self._schedule(self._end_trace(span, error))

    def log_event(self, trace_id: str, name: str, payload: dict[str, Any]) -> None:
        self._schedule(self._append_event(trace_id, name, payload))

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        raw = await self._redis.get(self._trace_key(trace_id))
        if not raw:
            return None
        return self._loads(raw)

    async def list_session_traces(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        ids = await self._redis.lrange(self._session_key(session_id), 0, max(limit - 1, 0))
        records: list[dict[str, Any]] = []
        for trace_id in ids:
            record = await self.get_trace(str(trace_id))
            if record:
                records.append(record)
        return records

    async def _start_trace(self, span: dict[str, Any]) -> None:
        trace_id = str(span["trace_id"])
        attrs = span.get("attributes") or {}
        now = span["started_at"]
        record = {
            "trace_id": trace_id,
            "otel_trace_id": trace_id if _is_otel_trace_id(trace_id) else None,
            "root_span_id": span.get("span_id"),
            "conversation_id": attrs.get("session_id"),
            "session_id": attrs.get("session_id"),
            "status": "running",
            "mode": attrs.get("mode") or "react",
            "intent": attrs.get("intent"),
            "started_at": now,
            "ended_at": None,
            "duration_ms": None,
            "error": None,
            "steps": [],
            "events": [
                {
                    "ts": now,
                    "name": "span.start",
                    "payload": self._safe_payload({"name": span.get("name"), "attributes": attrs}),
                }
            ],
        }
        await self._redis.set(self._trace_key(trace_id), self._dumps(record), ex=self._ttl)
        if attrs.get("session_id"):
            skey = self._session_key(str(attrs["session_id"]))
            await self._redis.lpush(skey, trace_id)
            await self._redis.ltrim(skey, 0, 49)
            await self._redis.expire(skey, self._ttl)

    async def _end_trace(self, span: Any, error: BaseException | None) -> None:
        if not isinstance(span, dict):
            return
        trace_id = str(span.get("trace_id") or "")
        if not trace_id:
            return
        record = await self._get_or_create(trace_id)
        ended_at = time.time()
        duration_ms = int((time.perf_counter() - float(span.get("_perf_start", time.perf_counter()))) * 1000)
        record.update(
            {
                "status": "failed" if error else "success",
                "ended_at": ended_at,
                "duration_ms": duration_ms,
                "error": str(error) if error else None,
            }
        )
        record["events"] = (record.get("events") or []) + [
            {
                "ts": ended_at,
                "name": "span.end",
                "payload": self._safe_payload({"error": str(error) if error else None, "duration_ms": duration_ms}),
            }
        ]
        record["events"] = record["events"][-self._max_events :]
        await self._save(trace_id, record)

    async def _append_event(self, trace_id: str, name: str, payload: dict[str, Any]) -> None:
        record = await self._get_or_create(trace_id)
        event = {"ts": time.time(), "name": name, "payload": self._safe_payload(payload)}
        record.setdefault("events", []).append(event)
        record["events"] = record["events"][-self._max_events :]

        step = self._event_to_step(name, payload, len(record.get("steps") or []) + 1)
        if step:
            record.setdefault("steps", []).append(step)
            record["steps"] = record["steps"][-self._max_steps :]
        await self._save(trace_id, record)

    async def _get_or_create(self, trace_id: str) -> dict[str, Any]:
        existing = await self.get_trace(trace_id)
        if existing:
            return existing
        now = time.time()
        return {
            "trace_id": trace_id,
            "otel_trace_id": trace_id if _is_otel_trace_id(trace_id) else None,
            "root_span_id": current_otel_span_id(),
            "conversation_id": None,
            "session_id": None,
            "status": "running",
            "mode": "react",
            "intent": None,
            "started_at": now,
            "ended_at": None,
            "duration_ms": None,
            "error": None,
            "steps": [],
            "events": [],
        }

    async def _save(self, trace_id: str, record: dict[str, Any]) -> None:
        await self._redis.set(self._trace_key(trace_id), self._dumps(record), ex=self._ttl)

    def _event_to_step(self, name: str, payload: dict[str, Any], index: int) -> dict[str, Any] | None:
        if name not in {"react.step", "plan_execute"}:
            return None
        phase = str(payload.get("phase") or payload.get("subtask_id") or name)
        status = str(payload.get("status") or ("error" if payload.get("error") else "success"))
        duration = payload.get("duration_ms", payload.get("tool_duration_ms"))
        return {
            "index": index,
            "phase": phase,
            "status": status,
            "action": payload.get("action") or payload.get("tool") or payload.get("subtask_id"),
            "action_input": payload.get("action_input") if isinstance(payload.get("action_input"), dict) else None,
            "observation": self._trim(payload.get("observation") or payload.get("result") or payload.get("output")),
            "duration_ms": int(duration) if isinstance(duration, (int, float)) else None,
            "error": self._trim(payload.get("error")),
            "span_id": current_otel_span_id(),
            "raw": self._safe_payload(payload),
        }

    def _schedule(self, coro: Any) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._guard(coro))
        except RuntimeError:
            logger.debug("Skip async trace write because no event loop is running")

    async def _guard(self, coro: Any) -> None:
        try:
            await coro
        except Exception as exc:  # pragma: no cover - observability must not break requests
            logger.warning("Redis trace write failed: {}", exc)

    def _safe_payload(self, payload: Any) -> Any:
        try:
            normalized = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
        except Exception:
            normalized = str(payload)
        return self._trim_nested(normalized)

    def _trim_nested(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._trim(value)
        if isinstance(value, list):
            return [self._trim_nested(v) for v in value[:50]]
        if isinstance(value, dict):
            return {str(k): self._trim_nested(v) for k, v in list(value.items())[:80]}
        return value

    def _trim(self, value: Any, max_chars: int = 4000) -> Any:
        if not isinstance(value, str):
            return value
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + "...[truncated]"

    def _loads(self, raw: str) -> dict[str, Any]:
        return json.loads(raw)

    def _dumps(self, record: dict[str, Any]) -> str:
        return json.dumps(record, ensure_ascii=False, default=str)

    def _trace_key(self, trace_id: str) -> str:
        return f"agent:trace:{trace_id}"

    def _session_key(self, session_id: str) -> str:
        return f"agent:session:{session_id}:traces"


def _is_otel_trace_id(value: str) -> bool:
    return len(value) == 32 and all(ch in "0123456789abcdef" for ch in value.lower())
