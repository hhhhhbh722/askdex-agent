# -*- coding: utf-8 -*-
"""文档 API：后台上传任务、列表、删除、RAG 检索。"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.wiring import embed_chunks, get_state, rag_search
from app.etl import ETLPipeline
from app.infrastructure.database.models import Document, DocumentChunk
from app.infrastructure.database import session as db_session
from app.infrastructure.database.session import get_async_session
from app.models.schemas import DocumentGroupUpdate, DocumentInfo, DocumentUploadResponse

router = APIRouter(tags=["documents"])

_upload_jobs: dict[str, dict[str, Any]] = {}
_reindex_jobs: dict[str, dict[str, Any]] = {}


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload(
    file: UploadFile = File(...),
    group: str = Form(default=""),
    parent_group: str = Form(default=""),
    child_group: str = Form(default=""),
    session: AsyncSession = Depends(get_async_session),
) -> DocumentUploadResponse:
    """单文件同步上传，保留给旧调用兼容。"""
    return await _ingest_document_bytes(
        raw=await file.read(),
        filename=file.filename or "unnamed",
        mime_type=file.content_type,
        group=group,
        parent_group=parent_group,
        child_group=child_group,
        session=session,
    )


@router.post("/documents/batch-upload")
async def batch_upload(
    files: list[UploadFile] = File(...),
    group: str = Form(default=""),
    parent_group: str = Form(default=""),
    child_group: str = Form(default=""),
) -> dict:
    """创建批量上传后台任务，立即返回 job_id。"""
    if not files:
        raise HTTPException(400, "请选择至少一个文件")

    group_path, normalized_parent, normalized_child = _normalize_group(
        group=group,
        parent_group=parent_group,
        child_group=child_group,
    )
    payloads = [{
        "raw": await file.read(),
        "filename": file.filename or "unnamed",
        "mime_type": file.content_type,
    } for file in files]

    job_id = str(uuid.uuid4())
    _upload_jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "total": len(payloads),
        "processed": 0,
        "success": 0,
        "failed": 0,
        "group": group_path,
        "parent_group": normalized_parent,
        "child_group": normalized_child,
        "current": "",
        "results": [],
    }
    asyncio.create_task(_run_batch_upload_job(
        job_id=job_id,
        payloads=payloads,
        group=group_path,
        parent_group=normalized_parent,
        child_group=normalized_child,
    ))
    return {"job_id": job_id, **_upload_jobs[job_id]}


@router.get("/documents/jobs/{job_id}")
async def get_upload_job(job_id: str) -> dict:
    """查询批量上传任务状态。"""
    job = _upload_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "上传任务不存在或服务已重启")
    return job


@router.post("/documents/reindex")
async def reindex_documents() -> dict:
    """Start a background job that rebuilds Milvus vectors from PostgreSQL."""
    state = get_state()
    if not state.get("milvus") or not state.get("settings"):
        raise HTTPException(503, "Milvus 未就绪，无法重建索引")
    if db_session.async_session_factory is None:
        raise HTTPException(503, "数据库 session 工厂未初始化")

    job_id = str(uuid.uuid4())
    _reindex_jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "documents": 0,
        "processed_documents": 0,
        "chunks": 0,
        "indexed_chunks": 0,
        "current": "",
        "failed": [],
    }
    asyncio.create_task(_run_reindex_job(job_id))
    return {"job_id": job_id, **_reindex_jobs[job_id]}


@router.get("/documents/reindex/jobs/{job_id}")
async def get_reindex_job(job_id: str) -> dict:
    job = _reindex_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "重建索引任务不存在或服务已重启")
    return job


async def _run_reindex_job(job_id: str) -> None:
    job = _reindex_jobs[job_id]
    job["status"] = "running"

    state = get_state()
    milvus = state.get("milvus")
    settings = state.get("settings")
    if not milvus or not settings or db_session.async_session_factory is None:
        job["status"] = "failed"
        job["failed"].append({"message": "Milvus 或数据库未就绪"})
        return

    batch: list[dict[str, Any]] = []
    batch_size = 10

    async with db_session.async_session_factory() as session:
        docs_r = await session.execute(select(Document).order_by(Document.created_at.asc()))
        docs = docs_r.scalars().all()
        job["documents"] = len(docs)

        for doc in docs:
            job["current"] = doc.filename
            chunks_r = await session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == doc.id)
                .order_by(DocumentChunk.chunk_index.asc())
            )
            chunks = chunks_r.scalars().all()
            if not chunks:
                job["processed_documents"] += 1
                continue

            group, parent_group, child_group = _groups_from_meta(doc.meta or {})
            job["chunks"] += len(chunks)

            for chunk in chunks:
                batch.append({
                    "id": chunk.vector_id or chunk.id,
                    "content": chunk.content,
                    "source": doc.filename,
                    "group": group,
                    "parent_group": parent_group,
                    "child_group": child_group,
                    "chunk_index": chunk.chunk_index,
                    "document_id": str(doc.id),
                })
                if len(batch) >= batch_size:
                    await _index_reindex_batch(job, batch)
                    batch.clear()
            job["processed_documents"] += 1

    if batch:
        await _index_reindex_batch(job, batch)

    job["current"] = ""
    job["status"] = "ready" if not job["failed"] else "partial"


async def _index_reindex_batch(job: dict[str, Any], batch: list[dict[str, Any]]) -> None:
    state = get_state()
    emb = state.get("embedding")
    milvus = state.get("milvus")
    settings = state.get("settings")
    if not emb or not milvus or not settings:
        job["failed"].append({"message": "Embedding 或 Milvus 未就绪"})
        return

    ids = [item["id"] for item in batch]
    try:
        vectors = await emb.aencode([item["content"] for item in batch])
        await milvus.delete(settings.milvus_collection_name, ids)
        await milvus.insert(settings.milvus_collection_name, vectors, batch)
        job["indexed_chunks"] += len(batch)
    except Exception as exc:
        logger.exception("批量重建索引失败: {}", exc)
        job["failed"].append({
            "document_id": "",
            "filename": "batch",
            "message": str(exc),
        })


async def _run_batch_upload_job(
    job_id: str,
    payloads: list[dict[str, Any]],
    group: str,
    parent_group: str,
    child_group: str,
) -> None:
    job = _upload_jobs[job_id]
    job["status"] = "running"

    if db_session.async_session_factory is None:
        job["status"] = "failed"
        job["failed"] = job["total"]
        job["processed"] = job["total"]
        job["results"] = [_failed_upload_result(
            filename=p["filename"],
            group=group,
            parent_group=parent_group,
            child_group=child_group,
            message="数据库 session 工厂未初始化",
        ) for p in payloads]
        return

    for payload in payloads:
        job["current"] = payload["filename"]
        async with db_session.async_session_factory() as session:
            try:
                uploaded = await _ingest_document_bytes(
                    raw=payload["raw"],
                    filename=payload["filename"],
                    mime_type=payload["mime_type"],
                    group=group,
                    parent_group=parent_group,
                    child_group=child_group,
                    session=session,
                )
                job["success"] += 1
                job["results"].append(uploaded.model_dump())
            except Exception as exc:
                logger.exception("批量上传失败: {}", payload["filename"])
                await session.rollback()
                job["failed"] += 1
                job["results"].append(_failed_upload_result(
                    filename=payload["filename"],
                    group=group,
                    parent_group=parent_group,
                    child_group=child_group,
                    message=str(exc),
                ))
            finally:
                job["processed"] += 1

    job["current"] = ""
    job["status"] = "ready" if job["failed"] == 0 else "partial"


async def _ingest_document_bytes(
    raw: bytes,
    filename: str,
    mime_type: str | None,
    group: str,
    parent_group: str,
    child_group: str,
    session: AsyncSession,
) -> DocumentUploadResponse:
    """保存文件、解析分块、写入 PostgreSQL 和 Milvus。"""
    upload_root = Path("uploads")
    upload_root.mkdir(parents=True, exist_ok=True)

    doc_id = str(uuid.uuid4())
    name = filename or "unnamed"
    doc_group, doc_parent_group, doc_child_group = _normalize_group(
        group=group,
        parent_group=parent_group,
        child_group=child_group,
    )
    dest = upload_root / f"{doc_id}_{name}"

    await asyncio.to_thread(dest.write_bytes, raw)
    etl = await ETLPipeline().run_bytes(raw, filename=name, mime_type=mime_type)

    doc = Document(
        id=doc_id,
        filename=name,
        mime_type=mime_type,
        storage_path=str(dest),
        status="ready",
        meta={
            "chunk_count": len(etl.chunks),
            "group": doc_group,
            "parent_group": doc_parent_group,
            "child_group": doc_child_group,
        },
    )
    session.add(doc)

    cids, ctexts = [], []
    for i, text in enumerate(etl.chunks):
        cid = str(uuid.uuid4())
        content = text[:65000]
        cids.append(cid)
        ctexts.append(content)
        session.add(DocumentChunk(
            id=cid,
            document_id=doc_id,
            chunk_index=i,
            content=content,
            vector_id=cid,
            meta={
                "group": doc_group,
                "parent_group": doc_parent_group,
                "child_group": doc_child_group,
                "source": name,
            },
        ))
    await session.flush()

    try:
        n = await embed_chunks(
            ctexts,
            cids,
            source=name,
            group=doc_group,
            parent_group=doc_parent_group,
            child_group=doc_child_group,
        )
        if n != len(etl.chunks):
            raise RuntimeError(f"向量入库不完整：{n}/{len(etl.chunks)}")
    except Exception:
        await session.rollback()
        raise

    await session.commit()
    logger.info("📄 {} [{}] → {} 块 → Milvus {}", name, doc_group or "未分组", len(etl.chunks), n)
    return DocumentUploadResponse(
        document_id=doc_id,
        filename=name,
        status="ready",
        chunk_count=len(etl.chunks),
        group=doc_group,
        parent_group=doc_parent_group,
        child_group=doc_child_group,
        message=f"上传成功，{n}/{len(etl.chunks)} 条向量入库",
    )


@router.get("/retrieve")
async def retrieve(
    q: str = Query(...),
    top_k: int = 5,
    group: str = Query(default=""),
    parent_group: str = Query(default=""),
    child_group: str = Query(default=""),
) -> dict:
    """RAG 向量检索，分组过滤前推到 Milvus。"""
    group_path, normalized_parent, normalized_child = _normalize_group(
        group=group,
        parent_group=parent_group,
        child_group=child_group,
    )
    filters = {
        "group": group_path if group.strip() else "",
        "parent_group": normalized_parent,
        "child_group": normalized_child,
    }
    results = await rag_search(q, top_k, filters=filters)
    return {"query": q, "results": results, "count": len(results)}


@router.delete("/documents/{doc_id}")
async def delete_doc(doc_id: str, session: AsyncSession = Depends(get_async_session)) -> dict:
    chunks_r = await session.execute(
        select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
    )
    chunks = chunks_r.scalars().all()

    doc = await session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")

    milvus = get_state().get("milvus")
    if milvus and chunks:
        ids = [c.vector_id or c.id for c in chunks]
        try:
            await milvus.delete(get_state()["settings"].milvus_collection_name, ids)
        except Exception as e:
            logger.warning("Milvus 删除失败: {}", e)
    await session.delete(doc)
    await session.commit()
    return {"status": "deleted", "id": doc_id}


@router.patch("/documents/{doc_id}/group")
async def update_doc_group(
    doc_id: str,
    payload: DocumentGroupUpdate,
    session: AsyncSession = Depends(get_async_session),
) -> dict:
    """更新文档分组，并同步重写 Milvus 元数据。"""
    doc = await session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")

    group, parent_group, child_group = _normalize_group(
        group=payload.group,
        parent_group=payload.parent_group,
        child_group=payload.child_group,
    )
    meta = dict(doc.meta or {})
    meta["group"] = group
    meta["parent_group"] = parent_group
    meta["child_group"] = child_group
    doc.meta = meta

    chunks_r = await session.execute(select(DocumentChunk).where(DocumentChunk.document_id == doc_id))
    chunks = chunks_r.scalars().all()
    for chunk in chunks:
        chunk_meta = dict(chunk.meta or {})
        chunk_meta["group"] = group
        chunk_meta["parent_group"] = parent_group
        chunk_meta["child_group"] = child_group
        chunk.meta = chunk_meta

    if chunks:
        milvus = get_state().get("milvus")
        if milvus:
            ids = [c.vector_id or c.id for c in chunks]
            texts = [c.content for c in chunks]
            try:
                await milvus.delete(get_state()["settings"].milvus_collection_name, ids)
                n = await embed_chunks(
                    texts,
                    ids,
                    source=doc.filename,
                    group=group,
                    parent_group=parent_group,
                    child_group=child_group,
                )
                if n != len(chunks):
                    raise RuntimeError(f"向量分组同步不完整：{n}/{len(chunks)}")
            except Exception:
                await session.rollback()
                raise

    await session.commit()
    return {
        "status": "ok",
        "id": doc_id,
        "group": group,
        "parent_group": parent_group,
        "child_group": child_group,
    }


@router.get("/evaluate")
async def evaluate_rag(q: str = Query(...), top_k: int = 5) -> dict:
    """RAG 评估：检索 Recall + MRR。"""
    from app.core.retrieval.pipeline import retrieval_pipeline
    from app.core.wiring import get_state
    state = get_state()
    emb, milvus = state.get("embedding"), state.get("milvus")
    if not emb or not milvus:
        return {"error": "Embedding 或 Milvus 未就绪"}

    results = await retrieval_pipeline(
        query=q, embedding=emb, milvus=milvus,
        collection=state["settings"].milvus_collection_name,
        top_k=top_k, hybrid=True, enable_hyde=True,
    )

    from app.core.evaluation import RAGEvaluator
    evaluator = RAGEvaluator()
    eval_result = evaluator.evaluate_retrieval([{
        "query": q,
        "relevant_ids": [r["id"] for r in results[:3]],
        "retrieved_ids": [r["id"] for r in results],
    }])

    return {
        "query": q,
        "results": [{"id": r["id"], "score": r["score"],
                      "content": r["content"][:200], "source": r.get("source", "")} for r in results],
        "metrics": {
            "recall_at_1": round(eval_result.recall_at_1, 4),
            "recall_at_3": round(eval_result.recall_at_3, 4),
            "recall_at_5": round(eval_result.recall_at_5, 4),
            "mrr": round(eval_result.mrr, 4),
        },
    }


@router.get("/documents", response_model=list[DocumentInfo])
async def list_docs(session: AsyncSession = Depends(get_async_session)) -> list[DocumentInfo]:
    r = await session.execute(select(Document).order_by(Document.created_at.desc()))
    docs = []
    for d in r.scalars().all():
        meta = d.meta or {}
        group, parent_group, child_group = _groups_from_meta(meta)
        docs.append(DocumentInfo(
            id=d.id,
            filename=d.filename,
            mime_type=d.mime_type,
            status=d.status,
            group=group,
            parent_group=parent_group,
            child_group=child_group,
            chunk_count=int(meta.get("chunk_count") or 0),
            created_at=d.created_at.isoformat() if d.created_at else None,
        ))
    return docs


def _failed_upload_result(
    filename: str,
    group: str,
    parent_group: str,
    child_group: str,
    message: str,
) -> dict:
    return {
        "document_id": "",
        "filename": filename,
        "status": "failed",
        "chunk_count": 0,
        "group": group,
        "parent_group": parent_group,
        "child_group": child_group,
        "message": message,
    }


def _normalize_group(
    group: str = "",
    parent_group: str = "",
    child_group: str = "",
) -> tuple[str, str, str]:
    """返回展示路径、一级分组、二级分组，兼容旧的单字段 group。"""
    parent = parent_group.strip()
    child = child_group.strip()
    path = group.strip()

    if not parent and path:
        if "/" in path:
            parent, child = [part.strip() for part in path.split("/", 1)]
        else:
            parent = path

    if parent and child:
        path = f"{parent} / {child}"
    elif parent:
        path = parent
    else:
        path = ""
        child = ""

    return path, parent, child


def _groups_from_meta(meta: dict) -> tuple[str, str, str]:
    return _normalize_group(
        group=str(meta.get("group") or ""),
        parent_group=str(meta.get("parent_group") or ""),
        child_group=str(meta.get("child_group") or ""),
    )
