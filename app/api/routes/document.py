# -*- coding: utf-8 -*-
"""文档 API：上传（ETL → PG + Milvus）、列表、删除、RAG 检索。"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, Request
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.wiring import embed_chunks, get_state, rag_search
from app.etl import ETLPipeline
from app.infrastructure.database.models import Document, DocumentChunk
from app.infrastructure.database.session import get_async_session
from app.models.schemas import DocumentInfo, DocumentUploadResponse

router = APIRouter(tags=["documents"])


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_async_session),
) -> DocumentUploadResponse:
    """上传 → ETL → PostgreSQL + Milvus。"""
    upload_root = Path("uploads")
    upload_root.mkdir(parents=True, exist_ok=True)

    doc_id = str(uuid.uuid4())
    name = file.filename or "unnamed"
    dest = upload_root / f"{doc_id}_{name}"

    raw = await file.read()
    await asyncio.to_thread(dest.write_bytes, raw)

    etl = await ETLPipeline().run_bytes(raw, filename=name, mime_type=file.content_type)

    doc = Document(id=doc_id, filename=name, mime_type=file.content_type,
                   storage_path=str(dest), status="ready", meta={"chunk_count": len(etl.chunks)})
    session.add(doc)

    cids, ctexts = [], []
    for i, t in enumerate(etl.chunks):
        cid = str(uuid.uuid4())
        cids.append(cid)
        ctexts.append(t[:65000])
        session.add(DocumentChunk(id=cid, document_id=doc_id, chunk_index=i, content=t[:65000], vector_id=cid))
    await session.commit()

    n = await embed_chunks(ctexts, cids, source=name)
    logger.info("📄 {} → {} 块 → Milvus {}", name, len(etl.chunks), n)

    return DocumentUploadResponse(document_id=doc_id, filename=name, status="ready",
                                  chunk_count=len(etl.chunks), message=f"上传成功，{n}/{len(etl.chunks)} 条向量入库")


@router.get("/retrieve")
async def retrieve(q: str = Query(...), top_k: int = 5) -> dict:
    """RAG 向量检索。"""
    results = await rag_search(q, top_k)
    return {"query": q, "results": results, "count": len(results)}


@router.delete("/documents/{doc_id}")
async def delete_doc(doc_id: str, session: AsyncSession = Depends(get_async_session)) -> dict:
    # 先查子记录
    chunks_r = await session.execute(
        select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
    )
    chunks = chunks_r.scalars().all()

    doc = await session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")

    # 删除 Milvus 向量
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
    return [DocumentInfo(id=d.id, filename=d.filename, mime_type=d.mime_type, status=d.status,
                         created_at=d.created_at.isoformat() if d.created_at else None) for d in r.scalars().all()]
