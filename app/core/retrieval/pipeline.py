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
    hybrid: bool = True, enable_hyde: bool = True,
    history: list[dict] | None = None,
    filters: dict | None = None,
    min_score: float = 0.15,
) -> list[dict]:
    """
    完整检索流水线：
    1. Query Rewrite（多轮对话上下文改写）
    2. 多查询生成（HyDE + 原文 + 改写）
    3. 混合检索（Dense + BM25）→ RRF 融合
    4. Reranker 精排
    5. 分数阈值过滤（min_score）

    :param query: 用户查询
    :param enable_hyde: 是否启用 HyDE 假设文档生成（推荐开启，显著提升检索精度）
    :param min_score: 最低分数阈值，低于此分数的结果被过滤（减少噪声）
    """

    # 1. Query Rewrite——多轮对话上下文改写
    rewritten = query
    if history and llm:
        rewritten = await rewrite_query(query, history, llm)

    queries = [query]
    if rewritten != query:
        queries.append(rewritten)

    # 2. HyDE：生成假设性答案，用答案的语义向量检索
    #    答案通常比问题更接近文档内容，可显著提升 Context Precision
    if enable_hyde and llm:
        try:
            hyde = await _generate_hyde(query, llm)
            if hyde and len(hyde) > 10:
                queries.append(hyde)
        except Exception:
            pass

    expr = _build_milvus_expr(filters or {})

    # 3. 每路检索：Dense + BM25 混合搜索
    all_lists: list[list[dict]] = []

    for q in queries:
        vec = (await embedding.aencode([q]))[0]
        try:
            results = await milvus.hybrid_search(
                collection, vec, q, top_k=top_k * 3,
                rerank_top_k=min(top_k * 10, 80), expr=expr)
            all_lists.append(results)
        except Exception as e:
            logger.warning("hybrid_search 失败: {}，降级 dense", e)
            results = await milvus.search(collection, vec, top_k=top_k * 3, expr=expr)
            all_lists.append(results)

    # 4. RRF 多路合并（原始 query + 改写 query + HyDE）
    merged = _rrf_fuse(all_lists, top_k=top_k * 3)

    # 5. Reranker 精排
    if reranker and len(merged) > top_k:
        merged = await reranker.rerank(query, merged[:min(top_k * 4, 30)], top_k)
    else:
        merged = merged[:top_k]

    # 6. 分数归一化：COSINE distance → 0-1 score
    for d in merged:
        d["score"] = max(0, 1 - d.get("distance", 0) / 2)
    merged.sort(key=lambda d: d["score"], reverse=True)

    # 7. 分数阈值过滤：去掉低分噪声
    before_filter = len(merged)
    merged = [d for d in merged if d["score"] >= min_score]
    if len(merged) < before_filter:
        logger.debug("分数阈值过滤: {} → {} (min_score={})", before_filter, len(merged), min_score)

    logger.info("检索完成 q={} results={} hybrid={} hyde={} reranker={} filtered={}/{}",
                query[:40], len(merged), hybrid, enable_hyde, bool(reranker),
                before_filter - len(merged), before_filter)
    return merged[:top_k]


def _build_milvus_expr(filters: dict) -> str | None:
    clauses = []
    for key in ("group", "parent_group", "child_group"):
        value = str(filters.get(key) or "").strip()
        if not value:
            continue
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        clauses.append(f'{key} == "{escaped}"')
    return " && ".join(clauses) if clauses else None


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
