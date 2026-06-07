# -*- coding: utf-8 -*-
"""Application-level retrieval services.

This module owns the GraphRAG fusion policy. Runtime wiring passes concrete
dependencies in, so retrieval behavior is kept away from app bootstrapping.
"""
from __future__ import annotations

from loguru import logger


async def rag_search(
    query: str,
    *,
    embedding,
    milvus,
    settings,
    llm=None,
    reranker=None,
    top_k: int = 5,
    filters: dict | None = None,
) -> list[dict]:
    """Retrieve with KG facts plus Milvus/BM25 hybrid results, then RRF fuse."""
    if not embedding or not milvus:
        return []

    from app.core.retrieval import retrieval_pipeline

    vector_results = await retrieval_pipeline(
        query=query,
        embedding=embedding,
        milvus=milvus,
        collection=settings.milvus_collection_name,
        llm=llm,
        reranker=reranker,
        top_k=top_k,
        hybrid=True,
        enable_hyde=True,
        filters=filters,
    )
    kg_results = await _kg_context_search(query, top_k=top_k, filters=filters)
    return _fuse_retrieval_results([kg_results, vector_results], top_k=top_k)


async def _kg_context_search(query: str, top_k: int, filters: dict | None = None) -> list[dict]:
    """Query PostgreSQL KG; filters are accepted for API compatibility."""
    _ = filters
    try:
        from app.core.kg.service import retrieve_kg_context
        from app.infrastructure.database import session as db_session

        if db_session.async_session_factory is None:
            return []
        async with db_session.async_session_factory() as session:
            return await retrieve_kg_context(session, query=query, top_k=top_k)
    except Exception as exc:
        logger.warning("KG 检索失败，跳过图谱分支: {}", exc)
        return []


def _fuse_retrieval_results(lists: list[list[dict]], top_k: int) -> list[dict]:
    """Fuse KG and vector/BM25 results with reciprocal rank fusion."""
    scores: dict[str, float] = {}
    best: dict[str, dict] = {}
    k = 60
    for weight, results in ((1.25, lists[0] if lists else []), (1.0, lists[1] if len(lists) > 1 else [])):
        for rank, item in enumerate(results or []):
            rid = str(item.get("document_id") or item.get("id"))
            scores[rid] = scores.get(rid, 0.0) + weight / (k + rank + 1)
            current = dict(item)
            current["fusion_sources"] = list(dict.fromkeys(
                [*(best.get(rid, {}).get("fusion_sources") or []), current.get("retrieval_source") or "vector_bm25"]
            ))
            if rid not in best or current.get("score", 0) > best[rid].get("score", 0):
                best[rid] = current

    ranked = sorted(scores, key=lambda rid: scores[rid], reverse=True)[:top_k]
    out: list[dict] = []
    for rid in ranked:
        item = best[rid]
        item["rrf_score"] = scores[rid]
        item["score"] = max(float(item.get("score") or 0), min(1.0, scores[rid] * 20))
        out.append(item)
    return out
