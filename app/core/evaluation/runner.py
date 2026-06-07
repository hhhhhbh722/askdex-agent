# -*- coding: utf-8 -*-
"""
RAGAS 评测执行器。

编排完整的 RAG 评估流水线：检索 → 生成 → 指标计算 → 报告。

支持三种评估模式：

- **retrieval_only**：仅评估检索阶段（Recall@K, MRR, Context Precision/Recall）
- **generation_only**：仅评估生成阶段（Faithfulness, Answer Relevancy, Answer Correctness），
  需要预先提供 contexts
- **full**：全流程评估，从 query 出发走完整 RAG 管线

核心类 ``RAGEvalRunner`` 依赖外部注入的检索函数和生成函数，
对现有 RAG 代码零侵入。

典型用法::

    from app.core.evaluation.llm_judge import LLMJudge
    from app.core.evaluation.runner import RAGEvalRunner
    from app.core.evaluation.testset import EvalTestSet
    from app.core.evaluation.reporter import EvalReport

    # 加载测试集
    testset = EvalTestSet.from_json("eval_testset.json")

    # 创建执行器
    runner = RAGEvalRunner(
        judge=LLMJudge(llm=my_llm),
        retrieve_fn=my_rag_search,        # async (query) -> list[dict]
        generate_fn=my_rag_generate,      # async (query, contexts) -> str
        embedding=my_embedding_api,
    )

    # 运行全流程评估
    report = await runner.run_full(testset)
    report.to_markdown("report.md")
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Awaitable

from loguru import logger

from .llm_judge import LLMJudge
from .ragas_metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    evaluate_retrieval_batch,
    faithfulness,
)
from .testset import EvalTestCase, EvalTestSet


# ---------------------------------------------------------------------------
# 单条用例的运行结果
# ---------------------------------------------------------------------------


@dataclass
class PerQueryResult:
    """
    单条测试用例的完整评估结果。

    属性:
        query: 原始查询
        contexts: 检索到的上下文（纯文本列表）
        answer: RAG 生成的答案
        metrics: 各指标得分字典，如 ``{"faithfulness": 0.85, "context_precision": 0.72}``
        ground_truth_answer: 标准答案（如有）
        latency_ms: 检索 + 生成的总耗时（毫秒）
    """

    query: str = ""
    contexts: list[str] = field(default_factory=list)
    answer: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    ground_truth_answer: str = ""
    relevant_ids: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# 评测执行器
# ---------------------------------------------------------------------------


class RAGEvalRunner:
    """
    RAGAS 评测执行器。

    编排检索→生成→评分全流程，支持三种评估模式。
    通过注入 ``retrieve_fn`` 和 ``generate_fn`` 解耦现有 RAG 管线。

    使用示例::

        async def my_retrieve(query: str) -> list[dict]:
            results = await retrieval_pipeline(query, ...)
            return results  # 每个 dict 包含 id, content, score 等

        async def my_generate(query: str, contexts: list[str]) -> str:
            response = await rag_generator.generate(query, contexts, [])
            return response.answer

        runner = RAGEvalRunner(
            judge=LLMJudge(llm=my_llm),
            retrieve_fn=my_retrieve,
            generate_fn=my_generate,
        )
    """

    def __init__(
        self,
        judge: LLMJudge,
        *,
        retrieve_fn: Callable[[str], Awaitable[list[dict]]] | None = None,
        generate_fn: Callable[[str, list[str]], Awaitable[str]] | None = None,
        embedding: Any = None,
        max_concurrency: int = 5,
    ) -> None:
        """
        :param judge: LLM 评判器实例
        :param retrieve_fn: 异步检索函数 ``(query: str) -> list[dict]``，
            每个 dict 至少包含 ``id`` 和 ``content``
        :param generate_fn: 异步生成函数 ``(query: str, contexts: list[str]) -> str``
        :param embedding: 可选的 EmbeddingAPI，用于 AnswerRelevancy 相似度计算
        :param max_concurrency: 并发评估的最大用例数（控制 LLM 调用频率）
        """
        self._judge = judge
        self._retrieve_fn = retrieve_fn
        self._generate_fn = generate_fn
        self._embedding = embedding
        self._max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

    # ------------------------------------------------------------------
    # 三种评估模式
    # ------------------------------------------------------------------

    async def run_retrieval_eval(
        self,
        testset: EvalTestSet,
    ) -> "EvalReport":
        """
        仅评估检索阶段。

        对每条用例调用 ``retrieve_fn`` 获取检索结果，
        计算 Recall@K, MRR, Context Precision, Context Recall。

        :param testset: 评测数据集
        :returns: EvalReport 评估报告
        """
        logger.info("开始检索评估: {} 条用例", len(testset))

        per_query: list[PerQueryResult] = []
        retrieval_cases: list[dict] = []  # 用于批量计算 Recall@K

        for tc in testset.test_cases:
            async with self._semaphore:
                result = await self._eval_retrieval_single(tc)
                per_query.append(result)

                if result.retrieved_ids:
                    retrieval_cases.append({
                        "query": tc.query,
                        "relevant_ids": tc.relevant_ids,
                        "retrieved_ids": result.retrieved_ids,
                    })

        # 批量计算检索命中率
        retrieval_metrics = evaluate_retrieval_batch(retrieval_cases)

        return EvalReport.from_per_query(
            testset_name=testset.name,
            per_query=per_query,
            retrieval_metrics=retrieval_metrics,
        )

    async def run_generation_eval(
        self,
        testset: EvalTestSet,
    ) -> "EvalReport":
        """
        仅评估生成阶段。

        测试用例中需预先包含 ``contexts``（由调用方提供），
        只调用 ``generate_fn`` 生成答案后计算生成指标。

        :param testset: 评测数据集（需预填 ground_truth 相关字段）
        :returns: EvalReport 评估报告
        """
        logger.info("开始生成评估: {} 条用例", len(testset))

        per_query: list[PerQueryResult] = []
        for tc in testset.test_cases:
            async with self._semaphore:
                result = await self._eval_generation_single(tc)
                per_query.append(result)

        return EvalReport.from_per_query(
            testset_name=testset.name,
            per_query=per_query,
        )

    async def run_full(
        self,
        testset: EvalTestSet,
    ) -> "EvalReport":
        """
        全流程评估：检索 + 生成 + 全量指标计算。

        对每条用例完整走一遍 RAG 管线，计算所有可用指标。

        :param testset: 评测数据集
        :returns: EvalReport 评估报告
        """
        logger.info("开始全流程评估: {} 条用例", len(testset))

        per_query: list[PerQueryResult] = []
        retrieval_cases: list[dict] = []

        for i, tc in enumerate(testset.test_cases):
            if i > 0 and i % 5 == 0:
                logger.info("全流程评估进度: {}/{}", i, len(testset))

            async with self._semaphore:
                result = await self._eval_full_single(tc)
                per_query.append(result)

                if result.retrieved_ids:
                    retrieval_cases.append({
                        "query": tc.query,
                        "relevant_ids": tc.relevant_ids,
                        "retrieved_ids": result.retrieved_ids,
                    })

        retrieval_metrics = evaluate_retrieval_batch(retrieval_cases)

        return EvalReport.from_per_query(
            testset_name=testset.name,
            per_query=per_query,
            retrieval_metrics=retrieval_metrics,
        )

    # ------------------------------------------------------------------
    # 单条用例评估
    # ------------------------------------------------------------------

    async def _eval_retrieval_single(self, tc: EvalTestCase) -> PerQueryResult:
        """仅检索评估：单条用例。"""
        result = PerQueryResult(
            query=tc.query,
            relevant_ids=tc.relevant_ids,
        )

        start = time.perf_counter()

        # 执行检索
        if self._retrieve_fn is None:
            result.error = "未配置 retrieve_fn"
            return result

        try:
            retrieved = await self._retrieve_fn(tc.query)
            result.contexts = [r.get("content", "") for r in retrieved]
            result.retrieved_ids = [r.get("id", "") for r in retrieved]
        except Exception as exc:
            logger.warning("检索失败 query={}: {}", tc.query[:40], exc)
            result.error = str(exc)

        result.latency_ms = (time.perf_counter() - start) * 1000

        # 计算检索指标
        try:
            if result.contexts:
                result.metrics["context_precision"] = await context_precision(
                    self._judge, tc.query, result.contexts,
                )
            else:
                result.metrics["context_precision"] = 0.0

            if tc.key_facts and result.contexts:
                result.metrics["context_recall"] = await context_recall(
                    self._judge, tc.query, result.contexts, tc.key_facts,
                )
            elif tc.ground_truth_contexts and result.contexts:
                result.metrics["context_recall"] = await context_recall(
                    self._judge, tc.query, result.contexts, tc.ground_truth_contexts,
                )
        except Exception as exc:
            logger.warning("检索指标计算失败 query={}: {}", tc.query[:40], exc)

        return result

    async def _eval_generation_single(self, tc: EvalTestCase) -> PerQueryResult:
        """仅生成评估：单条用例。"""
        result = PerQueryResult(
            query=tc.query,
            contexts=list(tc.ground_truth_contexts),
            ground_truth_answer=tc.ground_truth_answer,
        )

        if self._generate_fn is None:
            result.error = "未配置 generate_fn"
            return result

        start = time.perf_counter()

        # 执行生成
        try:
            result.answer = await self._generate_fn(tc.query, result.contexts)
        except Exception as exc:
            logger.warning("生成失败 query={}: {}", tc.query[:40], exc)
            result.error = str(exc)

        result.latency_ms = (time.perf_counter() - start) * 1000

        # 计算生成指标
        await self._compute_generation_metrics(tc, result)

        return result

    async def _eval_full_single(self, tc: EvalTestCase) -> PerQueryResult:
        """全流程评估：单条用例。"""
        result = PerQueryResult(
            query=tc.query,
            relevant_ids=tc.relevant_ids,
            ground_truth_answer=tc.ground_truth_answer,
        )

        total_start = time.perf_counter()

        # --- 阶段 1：检索 ---
        if self._retrieve_fn is None:
            result.error = "未配置 retrieve_fn"
            return result

        try:
            retrieved = await self._retrieve_fn(tc.query)
            result.contexts = [r.get("content", "") for r in retrieved]
            result.retrieved_ids = [r.get("id", "") for r in retrieved]
        except Exception as exc:
            logger.warning("检索失败 query={}: {}", tc.query[:40], exc)
            result.error = str(exc)
            return result

        retrieval_latency = (time.perf_counter() - total_start) * 1000

        # --- 阶段 2：生成 ---
        if self._generate_fn is not None and result.contexts:
            try:
                result.answer = await self._generate_fn(tc.query, result.contexts)
            except Exception as exc:
                logger.warning("生成失败 query={}: {}", tc.query[:40], exc)
                result.error = result.error or str(exc)

        result.latency_ms = (time.perf_counter() - total_start) * 1000

        # --- 阶段 3：指标计算 ---
        # 检索指标
        try:
            if result.contexts:
                result.metrics["context_precision"] = await context_precision(
                    self._judge, tc.query, result.contexts,
                )
            else:
                result.metrics["context_precision"] = 0.0

            # Context Recall：优先使用 key_facts，其次 ground_truth_contexts
            gt_sentences = tc.key_facts or tc.ground_truth_contexts
            if gt_sentences and result.contexts:
                result.metrics["context_recall"] = await context_recall(
                    self._judge, tc.query, result.contexts, gt_sentences,
                )
        except Exception as exc:
            logger.warning("检索指标计算失败 query={}: {}", tc.query[:40], exc)

        # 生成指标
        await self._compute_generation_metrics(tc, result)

        return result

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    async def _compute_generation_metrics(
        self,
        tc: EvalTestCase,
        result: PerQueryResult,
    ) -> None:
        """计算生成阶段的三个指标并填入 result.metrics。"""
        if not result.answer:
            return

        # Faithfulness
        try:
            if result.contexts:
                result.metrics["faithfulness"] = await faithfulness(
                    self._judge, result.answer, result.contexts,
                )
        except Exception as exc:
            logger.warning("faithfulness 计算失败 query={}: {}", tc.query[:40], exc)

        # Answer Relevancy
        try:
            result.metrics["answer_relevancy"] = await answer_relevancy(
                self._judge, tc.query, result.answer,
                embedding=self._embedding,
            )
        except Exception as exc:
            logger.warning("answer_relevancy 计算失败 query={}: {}", tc.query[:40], exc)

        # Answer Correctness（需要 Ground Truth）
        if tc.ground_truth_answer:
            try:
                result.metrics["answer_correctness"] = await answer_correctness(
                    self._judge, result.answer, tc.ground_truth_answer,
                )
            except Exception as exc:
                logger.warning("answer_correctness 计算失败 query={}: {}", tc.query[:40], exc)


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------


async def quick_eval(
    query: str,
    answer: str,
    contexts: list[str],
    judge: LLMJudge,
    *,
    ground_truth_answer: str = "",
    embedding: Any = None,
) -> dict[str, float]:
    """
    快速单条评估：对一条 (query, answer, contexts) 直接计算所有生成指标。

    适用于在线调试或监控场景。

    :param query: 用户查询
    :param answer: RAG 回答
    :param contexts: 检索到的上下文
    :param judge: LLM 评判器
    :param ground_truth_answer: 可选标准答案
    :param embedding: 可选 EmbeddingAPI
    :returns: 指标字典
    """
    metrics: dict[str, float] = {}

    if contexts:
        metrics["faithfulness"] = await faithfulness(judge, answer, contexts)
        metrics["context_precision"] = await context_precision(judge, query, contexts)

    metrics["answer_relevancy"] = await answer_relevancy(
        judge, query, answer, embedding=embedding,
    )

    if ground_truth_answer:
        metrics["answer_correctness"] = await answer_correctness(
            judge, answer, ground_truth_answer,
        )

    return metrics


# ======================================================================
# EvalReport（从 reporter.py 引用以避免循环依赖，实际定义在 reporter.py）
# ======================================================================

# 这里做前向声明，实际类型定义在 reporter.py
class EvalReport:
    """
    评估报告（前向声明）。

    实际实现在 ``app.core.evaluation.reporter`` 模块中。
    见 :class:`app.core.evaluation.reporter.EvalReport`。
    """

    @staticmethod
    def from_per_query(
        testset_name: str = "",
        per_query: list[PerQueryResult] | None = None,
        retrieval_metrics: dict[str, float] | None = None,
    ) -> Any:
        """从逐条结果构建报告（实际委托 reporter.py 实现）。"""
        from .reporter import EvalReport as RealEvalReport
        return RealEvalReport.from_per_query(
            testset_name=testset_name,
            per_query=per_query or [],
            retrieval_metrics=retrieval_metrics or {},
        )
