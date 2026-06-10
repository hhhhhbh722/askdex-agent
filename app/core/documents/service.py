# -*- coding: utf-8 -*-
"""Document ingestion, indexing, and grouping workflows."""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.kg.extractor import normalize_name
from app.core.wiring import embed_chunks, get_state
from app.etl import ETLPipeline
from app.infrastructure.database import session as db_session
from app.infrastructure.database.models import Document, DocumentChunk
from app.models.schemas import DocumentGroupUpdate, DocumentInfo, DocumentUploadResponse

_upload_jobs: dict[str, dict[str, Any]] = {}
_reindex_jobs: dict[str, dict[str, Any]] = {}
_VECTOR_REQUIRED_FIELDS = {"document_id", "entity_name", "normalized_entity_name"}


def get_upload_job(job_id: str) -> dict[str, Any] | None:
    return _upload_jobs.get(job_id)


def get_reindex_job(job_id: str) -> dict[str, Any] | None:
    return _reindex_jobs.get(job_id)


def start_batch_upload_job(
    payloads: list[dict[str, Any]],
    *,
    group: str = "",
    parent_group: str = "",
    child_group: str = "",
) -> dict[str, Any]:
    group_path, normalized_parent, normalized_child = normalize_group(
        group=group,
        parent_group=parent_group,
        child_group=child_group,
    )
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


def start_reindex_job() -> dict[str, Any]:
    state = get_state()
    if not state.get("milvus") or not state.get("settings"):
        raise RuntimeError("Milvus 未就绪，无法重建索引")
    if db_session.async_session_factory is None:
        raise RuntimeError("数据库 session 工厂未初始化")

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


async def vector_schema_status() -> dict[str, Any]:
    """Inspect Milvus collection fields required by GraphRAG v2."""
    state = get_state()
    milvus = state.get("milvus")
    settings = state.get("settings")
    if not milvus or not settings:
        raise RuntimeError("Milvus 未就绪，无法检测 schema")
    if not hasattr(milvus, "scalar_fields"):
        raise RuntimeError("当前 Milvus 客户端不支持 schema 检测")

    fields = await milvus.scalar_fields(settings.milvus_collection_name)
    missing = sorted(_VECTOR_REQUIRED_FIELDS - fields)
    return {
        "collection": settings.milvus_collection_name,
        "fields": sorted(fields),
        "required_fields": sorted(_VECTOR_REQUIRED_FIELDS),
        "missing_fields": missing,
        "needs_recreate": bool(missing),
    }


async def ingest_document_bytes(
    *,
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
    entity_name, normalized_entity_name = entity_names_from_filename(name)
    doc_group, doc_parent_group, doc_child_group = normalize_group(
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
            "entity_name": entity_name,
            "normalized_entity_name": normalized_entity_name,
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
                "entity_name": entity_name,
                "normalized_entity_name": normalized_entity_name,
            },
        ))
    await session.flush()

    try:
        indexed_count = await embed_chunks(
            ctexts,
            cids,
            source=name,
            document_id=doc_id,
            entity_name=entity_name,
            normalized_entity_name=normalized_entity_name,
            group=doc_group,
            parent_group=doc_parent_group,
            child_group=doc_child_group,
        )
        if indexed_count != len(etl.chunks):
            raise RuntimeError(f"向量入库不完整：{indexed_count}/{len(etl.chunks)}")
    except Exception:
        await session.rollback()
        raise

    await session.commit()
    logger.info("📄 {} [{}] → {} 块 → Milvus {}", name, doc_group or "未分组", len(etl.chunks), indexed_count)
    return DocumentUploadResponse(
        document_id=doc_id,
        filename=name,
        status="ready",
        chunk_count=len(etl.chunks),
        group=doc_group,
        parent_group=doc_parent_group,
        child_group=doc_child_group,
        message=f"上传成功，{indexed_count}/{len(etl.chunks)} 条向量入库",
    )


async def delete_document(doc_id: str, session: AsyncSession) -> bool:
    chunks_r = await session.execute(
        select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
    )
    chunks = chunks_r.scalars().all()

    doc = await session.get(Document, doc_id)
    if not doc:
        return False

    milvus = get_state().get("milvus")
    if milvus and chunks:
        ids = [c.vector_id or c.id for c in chunks]
        try:
            await milvus.delete(get_state()["settings"].milvus_collection_name, ids)
        except Exception as exc:
            logger.warning("Milvus 删除失败: {}", exc)
    await session.delete(doc)
    await session.commit()
    return True


async def update_document_group(
    doc_id: str,
    payload: DocumentGroupUpdate,
    session: AsyncSession,
) -> dict[str, Any] | None:
    doc = await session.get(Document, doc_id)
    if not doc:
        return None

    group, parent_group, child_group = normalize_group(
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
            entity_name, normalized_entity_name = entity_names_from_filename(doc.filename)
            try:
                await milvus.delete(get_state()["settings"].milvus_collection_name, ids)
                indexed_count = await embed_chunks(
                    texts,
                    ids,
                    source=doc.filename,
                    document_id=doc.id,
                    entity_name=entity_name,
                    normalized_entity_name=normalized_entity_name,
                    group=group,
                    parent_group=parent_group,
                    child_group=child_group,
                )
                if indexed_count != len(chunks):
                    raise RuntimeError(f"向量分组同步不完整：{indexed_count}/{len(chunks)}")
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


async def list_documents(session: AsyncSession) -> list[DocumentInfo]:
    r = await session.execute(select(Document).order_by(Document.created_at.desc()))
    docs = []
    for doc in r.scalars().all():
        meta = doc.meta or {}
        group, parent_group, child_group = groups_from_meta(meta)
        docs.append(DocumentInfo(
            id=doc.id,
            filename=doc.filename,
            mime_type=doc.mime_type,
            status=doc.status,
            group=group,
            parent_group=parent_group,
            child_group=child_group,
            chunk_count=int(meta.get("chunk_count") or 0),
            created_at=doc.created_at.isoformat() if doc.created_at else None,
        ))
    return docs


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

            group, parent_group, child_group = groups_from_meta(doc.meta or {})
            entity_name, normalized_entity_name = entity_names_from_filename(doc.filename)
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
                    "entity_name": entity_name,
                    "normalized_entity_name": normalized_entity_name,
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
                uploaded = await ingest_document_bytes(
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


def normalize_group(
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


def groups_from_meta(meta: dict) -> tuple[str, str, str]:
    return normalize_group(
        group=str(meta.get("group") or ""),
        parent_group=str(meta.get("parent_group") or ""),
        child_group=str(meta.get("child_group") or ""),
    )


def entity_names_from_filename(filename: str) -> tuple[str, str]:
    entity_name = normalize_name(Path(filename or "unnamed").stem)
    return entity_name, normalize_name(entity_name).lower()
