# -*- coding: utf-8 -*-
"""文档 API：上传、列表、删除、分组、RAG 检索。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.documents.service import (
    delete_document,
    get_reindex_job as get_reindex_job_state,
    get_upload_job as get_upload_job_state,
    ingest_document_bytes,
    list_documents,
    normalize_group,
    start_batch_upload_job,
    start_reindex_job,
    update_document_group,
    vector_schema_status,
)
from app.core.wiring import rag_search
from app.infrastructure.database.session import get_async_session
from app.models.schemas import DocumentGroupUpdate, DocumentInfo, DocumentUploadResponse

router = APIRouter(tags=["documents"])


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload(
    file: UploadFile = File(...),
    group: str = Form(default=""),
    parent_group: str = Form(default=""),
    child_group: str = Form(default=""),
    session: AsyncSession = Depends(get_async_session),
) -> DocumentUploadResponse:
    """单文件同步上传，保留给旧调用兼容。"""
    return await ingest_document_bytes(
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

    payloads = [{
        "raw": await file.read(),
        "filename": file.filename or "unnamed",
        "mime_type": file.content_type,
    } for file in files]
    return start_batch_upload_job(
        payloads=payloads,
        group=group,
        parent_group=parent_group,
        child_group=child_group,
    )


@router.get("/documents/jobs/{job_id}")
async def get_upload_job(job_id: str) -> dict:
    """查询批量上传任务状态。"""
    job = get_upload_job_state(job_id)
    if not job:
        raise HTTPException(404, "上传任务不存在或服务已重启")
    return job


@router.post("/documents/reindex")
async def reindex_documents() -> dict:
    """Start a background job that rebuilds Milvus vectors from PostgreSQL."""
    try:
        return start_reindex_job()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/documents/reindex/jobs/{job_id}")
async def get_reindex_job(job_id: str) -> dict:
    job = get_reindex_job_state(job_id)
    if not job:
        raise HTTPException(404, "重建索引任务不存在或服务已重启")
    return job


@router.get("/documents/vector-schema")
async def get_vector_schema() -> dict:
    try:
        return await vector_schema_status()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/retrieve")
async def retrieve(
    q: str = Query(...),
    top_k: int = 5,
    group: str = Query(default=""),
    parent_group: str = Query(default=""),
    child_group: str = Query(default=""),
) -> dict:
    """RAG 向量检索，分组过滤前推到 Milvus。"""
    group_path, normalized_parent, normalized_child = normalize_group(
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
    deleted = await delete_document(doc_id, session)
    if not deleted:
        raise HTTPException(404, "文档不存在")
    return {"status": "deleted", "id": doc_id}


@router.patch("/documents/{doc_id}/group")
async def update_doc_group(
    doc_id: str,
    payload: DocumentGroupUpdate,
    session: AsyncSession = Depends(get_async_session),
) -> dict:
    """更新文档分组，并同步重写 Milvus 元数据。"""
    result = await update_document_group(doc_id, payload, session)
    if not result:
        raise HTTPException(404, "文档不存在")
    return result


@router.get("/documents", response_model=list[DocumentInfo])
async def list_docs(session: AsyncSession = Depends(get_async_session)) -> list[DocumentInfo]:
    return await list_documents(session)
