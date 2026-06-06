# -*- coding: utf-8 -*-
"""轻量知识图谱 API。"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.kg.service import (
    build_document_kg,
    clear_kg,
    enrich_document_with_llm,
    entity_relations,
    graph_search,
    is_spirit_document,
    kg_stats,
    search_entities,
)
from app.core.wiring import get_state
from app.infrastructure.database import session as db_session
from app.infrastructure.database.models import Document
from app.infrastructure.database.session import get_async_session

router = APIRouter(tags=["knowledge-graph"])

_kg_rebuild_jobs: dict[str, dict[str, Any]] = {}
_kg_enrich_jobs: dict[str, dict[str, Any]] = {}


@router.get("/kg/stats")
async def stats(session: AsyncSession = Depends(get_async_session)) -> dict[str, Any]:
    return await kg_stats(session)


@router.get("/kg/search")
async def search(
    q: str = Query(default=""),
    type: str = Query(default=""),
    limit: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    entities = await search_entities(session, q=q, entity_type=type, limit=limit)
    return {"query": q, "type": type, "entities": entities, "count": len(entities)}


@router.get("/kg/query")
async def query(
    q: str = Query(...),
    limit: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    return await graph_search(session, q=q, limit=limit)


@router.get("/kg/graph")
async def graph(
    entity: str = Query(...),
    depth: int = Query(default=1, ge=1, le=2),
    limit: int = Query(default=120, ge=1, le=300),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    return await entity_relations(session, entity=entity, depth=depth, limit=limit)


@router.post("/kg/rebuild")
async def rebuild_kg() -> dict[str, Any]:
    if db_session.async_session_factory is None:
        raise HTTPException(503, "数据库 session 工厂未初始化")

    job_id = str(uuid.uuid4())
    _kg_rebuild_jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "documents": 0,
        "processed_documents": 0,
        "entities": 0,
        "relations": 0,
        "current": "",
        "failed": [],
    }
    asyncio.create_task(_run_rebuild_job(job_id))
    return {"job_id": job_id, **_kg_rebuild_jobs[job_id]}


@router.get("/kg/rebuild/jobs/{job_id}")
async def get_rebuild_job(job_id: str) -> dict[str, Any]:
    job = _kg_rebuild_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "KG 重建任务不存在或服务已重启")
    return job


@router.post("/kg/enrich")
async def enrich_kg(
    limit: int = Query(default=50, ge=1, le=1000),
) -> dict[str, Any]:
    if db_session.async_session_factory is None:
        raise HTTPException(503, "数据库 session 工厂未初始化")
    if not get_state().get("agent_llm"):
        raise HTTPException(503, "LLM 未就绪，无法执行 enrich")

    job_id = str(uuid.uuid4())
    _kg_enrich_jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "documents": 0,
        "processed_documents": 0,
        "relations": 0,
        "skipped": 0,
        "current": "",
        "failed": [],
    }
    asyncio.create_task(_run_enrich_job(job_id, limit=limit))
    return {"job_id": job_id, **_kg_enrich_jobs[job_id]}


@router.get("/kg/enrich/jobs/{job_id}")
async def get_enrich_job(job_id: str) -> dict[str, Any]:
    job = _kg_enrich_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "KG enrich 任务不存在或服务已重启")
    return job


async def _run_rebuild_job(job_id: str) -> None:
    job = _kg_rebuild_jobs[job_id]
    job["status"] = "running"
    assert db_session.async_session_factory is not None
    async with db_session.async_session_factory() as session:
        try:
            await clear_kg(session)
            docs_r = await session.execute(select(Document).order_by(Document.created_at.asc()))
            docs = [doc for doc in docs_r.scalars().all() if is_spirit_document(doc)]
            job["documents"] = len(docs)
            for doc in docs:
                job["current"] = doc.filename
                try:
                    result = await build_document_kg(session, doc)
                    job["entities"] += int(result.get("entities") or 0)
                    job["relations"] += int(result.get("relations") or 0)
                    job["processed_documents"] += 1
                    if job["processed_documents"] % 50 == 0:
                        await session.commit()
                except Exception as exc:
                    logger.exception("KG 重建失败: {}", doc.filename)
                    job["failed"].append({"id": doc.id, "filename": doc.filename, "error": str(exc)})
            await session.commit()
            job["status"] = "completed"
            job["current"] = ""
        except Exception as exc:
            await session.rollback()
            logger.exception("KG 重建任务失败")
            job["status"] = "failed"
            job["error"] = str(exc)


async def _run_enrich_job(job_id: str, limit: int) -> None:
    job = _kg_enrich_jobs[job_id]
    job["status"] = "running"
    llm = get_state().get("agent_llm")
    assert db_session.async_session_factory is not None
    async with db_session.async_session_factory() as session:
        try:
            docs = await _select_docs_for_enrich(session, limit=limit)
            job["documents"] = len(docs)
            for doc in docs:
                job["current"] = doc.filename
                try:
                    result = await enrich_document_with_llm(session, doc, llm)
                    if result.get("skipped"):
                        job["skipped"] += 1
                    job["relations"] += int(result.get("relations") or 0)
                    job["processed_documents"] += 1
                    await session.commit()
                except Exception as exc:
                    await session.rollback()
                    logger.exception("KG enrich 失败: {}", doc.filename)
                    job["failed"].append({"id": doc.id, "filename": doc.filename, "error": str(exc)})
            job["status"] = "completed"
            job["current"] = ""
        except Exception as exc:
            await session.rollback()
            logger.exception("KG enrich 任务失败")
            job["status"] = "failed"
            job["error"] = str(exc)


async def _select_docs_for_enrich(session: AsyncSession, limit: int) -> list[Document]:
    """选择尚未跑过 LLM enrich 的精灵图鉴文档。"""
    from app.infrastructure.database.models import KGRelation

    llm_doc_ids_r = await session.execute(
        select(KGRelation.source_document_id).where(KGRelation.extractor == "llm")
    )
    llm_doc_ids = {doc_id for doc_id in llm_doc_ids_r.scalars().all() if doc_id}

    docs_r = await session.execute(select(Document).order_by(Document.created_at.asc()))
    docs: list[Document] = []
    for doc in docs_r.scalars().all():
        if len(docs) >= limit:
            break
        if doc.id in llm_doc_ids:
            continue
        if not is_spirit_document(doc):
            continue
        docs.append(doc)
    return docs
