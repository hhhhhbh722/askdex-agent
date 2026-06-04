# -*- coding: utf-8 -*-
"""健康检查：存活、就绪、完整探测。"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.infrastructure.database.session import get_async_session

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/health/ready")
async def health_ready(session: AsyncSession = Depends(get_async_session)) -> dict[str, Any]:
    settings = get_settings()
    try:
        await session.execute(text("SELECT 1"))
        db = "up"
        status = "ready"
    except Exception as e:
        logger.warning("DB 就绪检查失败: {}", e)
        db = "down"
        status = "degraded"
    return {"status": status, "database": db, "app_env": settings.app_env}


@router.get("/health/full")
async def health_full(session: AsyncSession = Depends(get_async_session)) -> dict[str, Any]:
    """完整检查：API + PostgreSQL + Redis + Milvus。"""
    from app.core.wiring import get_state

    settings = get_settings()
    state = get_state()

    # DB
    try:
        await session.execute(text("SELECT 1"))
        db = "up"
    except Exception:
        db = "down"

    # Redis
    redis_client = state.get("redis")
    try:
        if redis_client:
            await redis_client.ping()
            redis_s = "up"
        else:
            redis_s = "disabled"
    except Exception:
        redis_s = "down"

    # Milvus
    milvus = state.get("milvus")
    if milvus:
        try:
            await milvus._ensure_connection()
            milvus_s = "up"
        except Exception:
            milvus_s = "down"
    else:
        milvus_s = "disabled"

    all_ok = all(s == "up" for s in [db, redis_s, milvus_s] if s != "disabled")
    return {
        "status": "healthy" if all_ok else "degraded",
        "api": "up",
        "database": db,
        "redis": redis_s,
        "milvus": milvus_s,
        "app_env": settings.app_env,
    }
