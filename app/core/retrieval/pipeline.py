# -*- coding: utf-8 -*-
"""检索流水线：多查询 → 混合检索(向量+BM25) → RRF融合 → Reranker精排。"""
from __future__ import annotations

from collections import defaultdict

from loguru import logger

from .query_rewriter import rewrite_query
from .reranker import Reranker

# RRF 平滑参数
RRF_K = 60


async def retrieval_pipeline(
    query: str, embedding, milvus, collection: str,
    llm=None, reranker: Reranker | None = None, top_k: int = 5,
    hybrid: bool = True, enable_hyde: bool = False,
    history: list[dict] | None = None,
) -> list[dict]:
    """
    完整检索流水线：
    1. Query Rewrite（多轮对话上下文改写）
    2. 多查询生成（HyDE + 原文 + 改写）
    3. 混合检索（Dense + BM25）→ RRF 融合
    4. Reranker 精排
    """

    # 1. Query Rewrite——多轮对话上下文改写
    rewritten = query
    if history and llm:
        rewritten = await rewrite_query(query, history, llm)

    queries = [query]
    if rewritten != query:
        queries.append(rewritten)

    # HyDE
    if enable_hyde and llm:
        try:
            hyde = await _generate_hyde(query, llm)
            if hyde and len(hyde) > 10:
                queries.append(hyde)
        except Exception:
            pass

    # 每路检索结果
    all_lists: list[list[dict]] = []

    for q in queries:
        vec = (await embedding.aencode([q]))[0]
        try:
            results = await milvus.hybrid_search(
                collection, vec, q, top_k=top_k * 3, rerank_top_k=min(top_k * 10, 80))
            all_lists.append(results)
        except Exception as e:
            logger.warning("hybrid_search 失败: {}，降级 dense", e)
            results = await milvus.search(collection, vec, top_k=top_k * 3)
            all_lists.append(results)

    # RRF 多路合并
    merged = _rrf_fuse(all_lists, top_k=top_k * 3)

    # Reranker 精排
    if reranker and len(merged) > top_k:
        merged = await reranker.rerank(query, merged[:min(top_k * 4, 30)], top_k)
    else:
        merged = merged[:top_k]

    # 将 distance 转为 0-1 score（COSINE distance 越小越好）
    for d in merged:
        d["score"] = max(0, 1 - d.get("distance", 0) / 2)
    merged.sort(key=lambda d: d["score"], reverse=True)

    logger.info("检索完成 q={} results={} hybrid={} hyde={} reranker={}",
                query[:40], len(merged), hybrid, enable_hyde, bool(reranker))
    return merged[:top_k]


def _rrf_fuse(lists: list[list[dict]], top_k: int) -> list[dict]:
    """倒数排名融合——不依赖原始分数，只用排名位置。"""
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, dict] = {}

    for lst in lists:
        for rank, doc in enumerate(lst):
            rid = doc["id"]
            scores[rid] += 1.0 / (RRF_K + rank + 1)
            if rid not in best or doc.get("distance", 99) < best[rid].get("distance", 99):
                best[rid] = doc

    ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]
    for rid in ranked:
        best[rid]["rrf_score"] = scores[rid]
    return [best[rid] for rid in ranked]


async def _generate_hyde(query: str, llm) -> str:
    prompt = f"请根据以下问题，用中文写一段简短的假设性回答（50-100字）：\n{query}\n回答："
    resp = await llm.acomplete([{"role": "user", "content": prompt}], temperature=0.3)
    return resp[:200] if resp else ""
