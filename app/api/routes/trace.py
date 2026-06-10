# -*- coding: utf-8 -*-
"""Agent trace inspection APIs."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.core.wiring import get_state

router = APIRouter(tags=["trace"])


@router.get("/traces")
async def list_traces(
    session_id: str = Query(..., description="Conversation/session id"),
    limit: int = Query(default=20, ge=1, le=50),
) -> dict[str, Any]:
    """List recent Agent traces for a session."""
    tracer = get_state().get("agent_tracer")
    if not tracer:
        raise HTTPException(503, "Agent tracer is not initialized")

    list_fn = getattr(tracer, "list_session_traces", None)
    if callable(list_fn):
        records = await list_fn(session_id, limit=limit)
        return {"session_id": session_id, "count": len(records), "traces": records}

    events = _memory_events(tracer, session_id=session_id, limit=limit)
    return {"session_id": session_id, "count": len(events), "traces": events, "storage": "memory"}


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: str) -> dict[str, Any]:
    """Get one Agent trace, including status, events and normalized steps."""
    record = await _load_trace(trace_id)
    if not record:
        raise HTTPException(404, "Trace not found")
    return record


@router.get("/traces/{trace_id}/steps")
async def get_trace_steps(trace_id: str) -> dict[str, Any]:
    """Return normalized Agent steps for UI step replay."""
    record = await _load_trace(trace_id)
    if not record:
        raise HTTPException(404, "Trace not found")
    return {
        "trace_id": trace_id,
        "status": record.get("status"),
        "mode": record.get("mode"),
        "steps": record.get("steps") or [],
    }


@router.get("/traces/{trace_id}/replay")
async def replay_trace(trace_id: str) -> dict[str, Any]:
    """Return ordered events and steps for replay/debug panels."""
    record = await _load_trace(trace_id)
    if not record:
        raise HTTPException(404, "Trace not found")
    return {
        "trace_id": trace_id,
        "status": record.get("status"),
        "started_at": record.get("started_at"),
        "ended_at": record.get("ended_at"),
        "duration_ms": record.get("duration_ms"),
        "steps": record.get("steps") or [],
        "events": record.get("events") or [],
    }


async def _load_trace(trace_id: str) -> dict[str, Any] | None:
    tracer = get_state().get("agent_tracer")
    if not tracer:
        raise HTTPException(503, "Agent tracer is not initialized")

    get_fn = getattr(tracer, "get_trace", None)
    if callable(get_fn):
        return await get_fn(trace_id)

    events = [
        event
        for event in getattr(tracer, "events", [])
        if event.get("trace_id") == trace_id or (event.get("span") or {}).get("trace_id") == trace_id
    ]
    if not events:
        return None
    return {"trace_id": trace_id, "status": "memory_only", "events": events, "steps": []}


def _memory_events(tracer: Any, session_id: str, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in reversed(getattr(tracer, "events", [])):
        trace_id = event.get("trace_id") or (event.get("span") or {}).get("trace_id")
        if not trace_id or trace_id in seen:
            continue
        attrs = (event.get("span") or {}).get("attributes") or {}
        if attrs.get("session_id") and attrs.get("session_id") != session_id:
            continue
        seen.add(trace_id)
        records.append({"trace_id": trace_id, "status": "memory_only", "events": [event], "steps": []})
        if len(records) >= limit:
            break
    return records
