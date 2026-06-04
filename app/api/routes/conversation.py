# -*- coding: utf-8 -*-
"""对话历史 API：保存、列表、详情与删除。"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.models import Conversation, Message as DBMessage
from app.infrastructure.database.session import get_async_session

router = APIRouter(tags=["conversations"])


@router.post("/conversations")
async def save_conversation(
    payload: dict[str, Any],
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """保存完整对话（含消息列表）到数据库。"""
    conv_id = payload.get("id") or str(uuid.uuid4())
    title = payload.get("title", "未命名对话")
    msgs = payload.get("messages", [])

    # 检查是否已存在
    existing = await session.get(Conversation, conv_id)
    if existing:
        existing.title = title
        # 删除旧消息再重新写入
        await session.execute(
            delete(DBMessage).where(DBMessage.conversation_id == conv_id)
        )
    else:
        session.add(Conversation(id=conv_id, title=title, meta={}))

    for i, m in enumerate(msgs):
        session.add(
            DBMessage(
                id=str(uuid.uuid4()),
                conversation_id=conv_id,
                role=m.get("role", "user"),
                content=m.get("content", ""),
                meta={"index": i},
            )
        )

    await session.commit()
    return {"id": conv_id, "title": title, "message_count": len(msgs), "status": "saved"}


@router.get("/conversations")
async def list_conversations(
    session: AsyncSession = Depends(get_async_session),
) -> list[dict[str, Any]]:
    """列出所有对话会话（摘要，不包含消息详情）。"""
    result = await session.execute(
        select(Conversation).order_by(Conversation.updated_at.desc())
    )
    rows = result.scalars().all()
    out = []
    for c in rows:
        # 统计消息数
        count_result = await session.execute(
            select(DBMessage).where(DBMessage.conversation_id == c.id)
        )
        msg_count = len(count_result.scalars().all())
        first_msg = ""
        for m in count_result.scalars():
            if m.content:
                first_msg = m.content[:100]
                break
        mode = c.meta.get("mode", "react") if c.meta else "react"
        out.append({
            "id": c.id,
            "title": c.title or "未命名对话",
            "preview": first_msg,
            "message_count": msg_count,
            "mode": mode,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        })
    return out


@router.get("/conversations/{conv_id}")
async def get_conversation(
    conv_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """获取单个对话的完整内容（含所有消息）。"""
    conv = await session.get(Conversation, conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")

    result = await session.execute(
        select(DBMessage)
        .where(DBMessage.conversation_id == conv_id)
        .order_by(DBMessage.created_at.asc())
    )
    msgs = result.scalars().all()
    return {
        "id": conv.id,
        "title": conv.title,
        "mode": conv.meta.get("mode", "react") if conv.meta else "react",
        "messages": [
            {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat() if m.created_at else None}
            for m in msgs
        ],
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
    }


@router.delete("/conversations/{conv_id}")
async def delete_conversation(
    conv_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, str]:
    """删除对话及其所有消息。"""
    conv = await session.get(Conversation, conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")
    await session.delete(conv)
    await session.commit()
    return {"status": "deleted", "id": conv_id}
