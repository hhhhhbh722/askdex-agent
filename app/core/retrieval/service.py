# -*- coding: utf-8 -*-
"""Application-level retrieval services.

This module owns the GraphRAG fusion policy. Runtime wiring passes concrete
dependencies in, so retrieval behavior is kept away from app bootstrapping.
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from app.core.kg.extractor import normalize_name


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

    kg_results = await _kg_context_search(query, top_k=max(top_k * 2, 10), filters=filters)
    planned_filters = _build_kg_planned_filters(filters, kg_results)

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
        filters=planned_filters,
    )
    if not vector_results and planned_filters is not filters:
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
    return _fuse_retrieval_results([kg_results[:top_k], vector_results], top_k=top_k)


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


def _build_kg_planned_filters(filters: dict | None, kg_results: list[dict]) -> dict | None:
    if not kg_results:
        return filters

    planned = dict(filters or {})
    document_ids = list(dict.fromkeys(
        str(item.get("document_id") or "").strip()
        for item in kg_results
        if str(item.get("document_id") or "").strip()
    ))
    sources = list(dict.fromkeys(
        str(item.get("source") or "").strip()
        for item in kg_results
        if str(item.get("source") or "").strip() and item.get("source") != "knowledge_graph"
    ))
    entity_names = list(dict.fromkeys(_entity_name_from_source(source) for source in sources))
    entity_names = [name for name in entity_names if name]
    normalized_entity_names = list(dict.fromkeys(normalize_name(name).lower() for name in entity_names if name))

    if document_ids and not planned.get("document_ids"):
        planned["document_ids"] = document_ids[:8]
    if sources and not planned.get("sources"):
        planned["sources"] = sources[:8]
    if entity_names and not planned.get("entity_names"):
        planned["entity_names"] = entity_names[:8]
    if normalized_entity_names and not planned.get("normalized_entity_names"):
        planned["normalized_entity_names"] = normalized_entity_names[:8]
    planned["kg_planned"] = True
    return planned


def _entity_name_from_source(source: str) -> str:
    return normalize_name(Path(source).stem)
