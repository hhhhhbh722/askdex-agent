# -*- coding: utf-8 -*-
"""RAG 质量评估：检索命中率、答案忠实度、上下文相关性。"""
from __future__ import annotations

from dataclasses import dataclass, field
from loguru import logger


@dataclass
class EvalResult:
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0                    # Mean Reciprocal Rank
    faithfulness: float = 0.0            # 答案忠实度（基于 LLM 判定）
    context_relevance: float = 0.0       # 上下文相关性
    total_queries: int = 0
    details: list[dict] = field(default_factory=list)


class RAGEvaluator:
    """RAG 评估器：检索质量 + 答案质量。"""

    def __init__(self, llm=None):
        self._llm = llm

    def evaluate_retrieval(self, test_cases: list[dict]) -> EvalResult:
        """
        test_cases: [{"query": "问题", "relevant_ids": ["doc1", "doc2"], "retrieved_ids": ["doc1", "doc3"]}]
        """
        total = len(test_cases)
        if total == 0:
            return EvalResult()

        r1_hits = r3_hits = r5_hits = 0
        reciprocal_ranks = []

        for tc in test_cases:
            rel = set(tc["relevant_ids"])
            ret = tc["retrieved_ids"]

            if ret and ret[0] in rel: r1_hits += 1
            if any(r in rel for r in ret[:3]): r3_hits += 1
            if any(r in rel for r in ret[:5]): r5_hits += 1

            for rank, rid in enumerate(ret, 1):
                if rid in rel:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return EvalResult(
            recall_at_1=r1_hits / total,
            recall_at_3=r3_hits / total,
            recall_at_5=r5_hits / total,
            mrr=sum(reciprocal_ranks) / total,
            total_queries=total,
        )

    async def evaluate_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """LLM 判定答案是否忠实于检索上下文（0-1）。"""
        if not self._llm or not contexts:
            return 0.5
        prompt = (
            "你是严格的事实核查员。判断以下回答是否完全基于提供的上下文，没有编造信息。\n\n"
            f"上下文：\n" + "\n".join(f"[{i+1}] {c[:200]}" for i, c in enumerate(contexts)) +
            f"\n\n回答：{answer[:500]}\n\n"
            "请只输出一个 0-1 之间的数字，表示忠实度（1=完全基于上下文，0=严重编造）："
        )
        try:
            resp = await self._llm.acomplete([{"role": "user", "content": prompt}], temperature=0.0)
            score = float(resp.strip()[:5])
            return max(0, min(1, score))
        except Exception as e:
            logger.warning("Faithfulness 评估失败: {}", e)
            return 0.5

    async def evaluate_all(self, test_cases: list[dict], contexts_map: dict[str, list[str]]) -> EvalResult:
        """完整评估：检索质量 + 答案忠实度。"""
        result = self.evaluate_retrieval(test_cases)

        faith_scores = []
        for tc in test_cases:
            answer = tc.get("answer", "")
            query = tc.get("query", "")
            ctxs = contexts_map.get(query, [])
            if answer and ctxs:
                faith_scores.append(await self.evaluate_faithfulness(answer, ctxs))

        if faith_scores:
            result.faithfulness = sum(faith_scores) / len(faith_scores)
        result.details = [{"query": tc["query"], "retrieved": tc.get("retrieved_ids", []),
                           "relevant": tc.get("relevant_ids", [])} for tc in test_cases]
        logger.info("RAG eval: R@1={:.2%} R@3={:.2%} MRR={:.3f} Faith={:.2f}",
                    result.recall_at_1, result.recall_at_3, result.mrr, result.faithfulness)
        return result
