# -*- coding: utf-8 -*-
"""
RAG 质量评估：检索命中率、答案忠实度、上下文相关性。

此模块保留原有的 ``EvalResult`` 和 ``RAGEvaluator`` 接口以保持向后兼容。
底层实现已迁移到 ``ragas_metrics`` 和 ``llm_judge`` 模块，
新代码推荐直接使用更丰富的独立指标函数。

兼容映射::

    RAGEvaluator.evaluate_retrieval()   → ragas_metrics.evaluate_retrieval_batch()
    RAGEvaluator.evaluate_faithfulness()→ ragas_metrics.faithfulness()
    RAGEvaluator.evaluate_all()         → runner.RAGEvalRunner.run_full()
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger


@dataclass
class EvalResult:
    """
    评估结果（兼容保留）。

    属性:
        recall_at_1/3/5: 召回命中率
        mrr: 平均倒数排名
        faithfulness: 答案忠实度（LLM 判定，0-1）
        context_relevance: 上下文相关性
        context_precision: 上下文精度（新指标）
        context_recall: 上下文召回（新指标）
        answer_relevancy: 答案切题度（新指标）
        answer_correctness: 答案正确性（新指标）
        total_queries: 总评估用例数
        details: 逐条详情
    """

    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    faithfulness: float = 0.0
    context_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_relevancy: float = 0.0
    answer_correctness: float = 0.0
    total_queries: int = 0
    details: list[dict] = field(default_factory=list)


class RAGEvaluator:
    """
    RAG 评估器（兼容保留）。

    原有方法保持不变，同时新增桥接方法调用新的指标实现。
    新代码推荐直接使用 ``ragas_metrics`` 中的独立函数。

    使用示例::

        evaluator = RAGEvaluator(llm=my_llm)

        # 原有方法
        result = evaluator.evaluate_retrieval(test_cases)

        # 新方法（需要 LLM）
        faith = await evaluator.evaluate_faithfulness(answer, contexts)
    """

    def __init__(self, llm=None):
        """
        :param llm: 实现 ``acomplete(messages, temperature) -> str`` 的 LLM 对象
        """
        self._llm = llm

    # ------------------------------------------------------------------
    # 检索评估（无 LLM 依赖）
    # ------------------------------------------------------------------

    def evaluate_retrieval(self, test_cases: list[dict]) -> EvalResult:
        """
        计算检索命中率指标（Recall@K, MRR）。

        此方法不依赖 LLM，纯数学计算。

        :param test_cases: 列表，每项需包含:
            - ``query``: 查询问题
            - ``relevant_ids``: 标注的相关文档 ID 列表
            - ``retrieved_ids``: 系统检索返回的文档 ID 列表（按排名）
        :returns: EvalResult，包含 Recall@1/3/5 和 MRR
        """
        # 委托给新的独立函数
        from .ragas_metrics import evaluate_retrieval_batch

        metrics = evaluate_retrieval_batch(test_cases)
        return EvalResult(
            recall_at_1=metrics["recall_at_1"],
            recall_at_3=metrics["recall_at_3"],
            recall_at_5=metrics["recall_at_5"],
            mrr=metrics["mrr"],
            total_queries=len(test_cases),
        )

    # ------------------------------------------------------------------
    # 答案忠实度（LLM 依赖）
    # ------------------------------------------------------------------

    async def evaluate_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """
        LLM 判定答案是否忠实于检索上下文（0-1）。

        此方法委托给新的 ``ragas_metrics.faithfulness()``，
        使用 claims-based 方法进行更精确的评估。

        :param answer: RAG 生成的答案
        :param contexts: 检索到的上下文列表
        :returns: 忠实度分数 0-1
        """
        # 尝试使用新实现（claims-based）
        if self._llm is not None:
            try:
                from .llm_judge import LLMJudge
                from .ragas_metrics import faithfulness

                judge = LLMJudge(llm=self._llm)
                return await faithfulness(judge, answer, contexts)
            except Exception as exc:
                logger.warning("faithfulness 新实现调用失败，降级旧方法: {}", exc)

        # 降级：原有的简单 prompt 方法
        if not self._llm or not contexts:
            return 0.5
        prompt = (
            "你是严格的事实核查员。判断以下回答是否完全基于提供的上下文，没有编造信息。\n\n"
            "上下文：\n" + "\n".join(f"[{i+1}] {c[:200]}" for i, c in enumerate(contexts))
            + f"\n\n回答：{answer[:500]}\n\n"
            "请只输出一个 0-1 之间的数字，表示忠实度（1=完全基于上下文，0=严重编造）："
        )
        try:
            resp = await self._llm.acomplete(
                [{"role": "user", "content": prompt}], temperature=0.0
            )
            score = float(resp.strip()[:5])
            return max(0, min(1, score))
        except Exception as e:
            logger.warning("Faithfulness 评估失败: {}", e)
            return 0.5

    # ------------------------------------------------------------------
    # 上下文相关性（新指标桥接）
    # ------------------------------------------------------------------

    async def evaluate_context_precision(self, query: str, contexts: list[str]) -> float:
        """
        评估上下文精度：检索文档是否与问题相关（位置加权）。

        新方法，需要 LLM。

        :param query: 用户查询
        :param contexts: 检索到的上下文列表
        :returns: Context Precision 0-1
        """
        if self._llm is None:
            return 0.5
        from .llm_judge import LLMJudge
        from .ragas_metrics import context_precision

        judge = LLMJudge(llm=self._llm)
        return await context_precision(judge, query, contexts)

    async def evaluate_context_recall(
        self,
        query: str,
        contexts: list[str],
        ground_truth_sentences: list[str],
    ) -> float:
        """
        评估上下文召回：Ground Truth 信息是否被检索覆盖。

        新方法，需要 LLM + Ground Truth。

        :param query: 用户查询
        :param contexts: 检索到的上下文列表
        :param ground_truth_sentences: 标准答案关键句式列表
        :returns: Context Recall 0-1
        """
        if self._llm is None:
            return 0.5
        from .llm_judge import LLMJudge
        from .ragas_metrics import context_recall

        judge = LLMJudge(llm=self._llm)
        return await context_recall(judge, query, contexts, ground_truth_sentences)

    async def evaluate_answer_relevancy(self, query: str, answer: str) -> float:
        """
        评估答案相关性：回答是否切题。

        新方法，需要 LLM。

        :param query: 用户查询
        :param answer: RAG 生成的答案
        :returns: Answer Relevancy 0-1
        """
        if self._llm is None:
            return 0.5
        from .llm_judge import LLMJudge
        from .ragas_metrics import answer_relevancy

        judge = LLMJudge(llm=self._llm)
        return await answer_relevancy(judge, query, answer)

    # ------------------------------------------------------------------
    # 完整评估（兼容保留）
    # ------------------------------------------------------------------

    async def evaluate_all(
        self,
        test_cases: list[dict],
        contexts_map: dict[str, list[str]],
    ) -> EvalResult:
        """
        完整评估：检索质量 + 答案忠实度。

        这是兼容保留方法。如需完整的 6 指标评估，
        请使用 ``RAGEvalRunner``。

        :param test_cases: 测试用例列表
        :param contexts_map: ``query → list[context_text]`` 映射
        :returns: EvalResult
        """
        result = self.evaluate_retrieval(test_cases)

        # 答案忠实度
        faith_scores = []
        for tc in test_cases:
            answer = tc.get("answer", "")
            query = tc.get("query", "")
            ctxs = contexts_map.get(query, [])
            if answer and ctxs:
                faith_scores.append(await self.evaluate_faithfulness(answer, ctxs))

        if faith_scores:
            result.faithfulness = sum(faith_scores) / len(faith_scores)

        result.details = [
            {
                "query": tc["query"],
                "retrieved": tc.get("retrieved_ids", []),
                "relevant": tc.get("relevant_ids", []),
            }
            for tc in test_cases
        ]

        logger.info(
            "RAG eval: R@1={:.2%} R@3={:.2%} MRR={:.3f} Faith={:.2f}",
            result.recall_at_1,
            result.recall_at_3,
            result.mrr,
            result.faithfulness,
        )
        return result
