# -*- coding: utf-8 -*-
"""Reranker：Cross-Encoder 精排（DashScope API）。"""
from __future__ import annotations

import httpx
from loguru import logger


class Reranker:
    """使用 DashScope rerank API 对候选文档重排序。"""

    def __init__(self, api_key: str, model: str = "gte-rerank"):
        self.api_key = api_key
        self.model = model
        self._base = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    async def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        """
        documents: [{"id": ..., "content": ..., "score": ...}]
        返回: 重排序后的 top_k 结果，分数归一化到 0-1。
        """
        if len(documents) <= top_k:
            # 不需要精排，直接按原始分数排序返回
            return sorted(documents, key=lambda d: d.get("score", 0), reverse=True)[:top_k]

        texts = [d["content"][:1024] for d in documents]
        try:
            scores = await self._call_api(query, texts)
            # 合并分数
            for d, s in zip(documents, scores):
                d["rerank_score"] = s
            ranked = sorted(documents, key=lambda d: d.get("rerank_score", 0), reverse=True)
            return ranked[:top_k]
        except Exception as e:
            logger.warning("Reranker 失败，降级为原始排序: {}", e)
            return sorted(documents, key=lambda d: d.get("score", 0), reverse=True)[:top_k]

    async def _call_api(self, query: str, documents: list[str]) -> list[float]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": {"query": query, "documents": documents},
                   "parameters": {"top_n": len(documents)}}
        async with httpx.AsyncClient() as c:
            resp = await c.post(self._base, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("output", {}).get("results", [])
            return [float(r.get("relevance_score", 0.5)) for r in results]
