# -*- coding: utf-8 -*-
"""
RAGAS 评估模块。

提供完整的 RAG 系统质量评估能力，覆盖检索和生成两个阶段。

模块结构::

    llm_judge.py     — LLMJudge：统一的 LLM 评判器（打分、分类、陈述拆分、反向提问）
    testset.py       — EvalTestSet / EvalTestCase：评测数据集 Schema + TestSetGenerator 自动生成
    ragas_metrics.py — 6 大 RAGAS 指标（Context Precision/Recall, Faithfulness, Answer Relevancy/Correctness）
    runner.py        — RAGEvalRunner：评测执行器（检索→生成→评分全流程）
    reporter.py      — EvalReport：报告生成（JSON / Markdown / 分析）

快速开始::

    from app.core.evaluation import (
        LLMJudge,
        EvalTestSet,
        TestSetGenerator,
        RAGEvalRunner,
        EvalReport,
    )

    # 1. 创建评判器
    judge = LLMJudge(llm=my_llm)

    # 2. 加载或生成测试集
    testset = EvalTestSet.from_json("my_testset.json")

    # 3. 运行评估
    runner = RAGEvalRunner(
        judge=judge,
        retrieve_fn=my_retrieve_func,
        generate_fn=my_generate_func,
    )
    report = await runner.run_full(testset)

    # 4. 导出报告
    report.to_markdown("report.md")
"""

# 评判器
from .llm_judge import LLMJudge, JudgeLLMProtocol

# 测试数据集
from .testset import EvalTestCase, EvalTestSet, TestSetGenerator

# 指标
from .ragas_metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    evaluate_retrieval_batch,
    faithfulness,
    mrr,
    recall_at_k,
)

# 执行器
from .runner import PerQueryResult, RAGEvalRunner, quick_eval

# 报告
from .reporter import EvalReport

# 保留原有类以保持向后兼容
from .metrics import EvalResult, RAGEvaluator

__all__ = [
    # 评判器
    "LLMJudge",
    "JudgeLLMProtocol",
    # 测试集
    "EvalTestCase",
    "EvalTestSet",
    "TestSetGenerator",
    # 指标（独立函数）
    "recall_at_k",
    "mrr",
    "evaluate_retrieval_batch",
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    # 执行器
    "RAGEvalRunner",
    "PerQueryResult",
    "quick_eval",
    # 报告
    "EvalReport",
    # 向后兼容
    "EvalResult",
    "RAGEvaluator",
]
