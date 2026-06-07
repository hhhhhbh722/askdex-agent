# RAGAS 评估模块

RAG（检索增强生成）系统的全面质量评估，覆盖检索和生成两个阶段。

## 模块结构

```
evaluation/
├── __init__.py       # 模块统一导出
├── llm_judge.py      # LLM 评判器（打分/分类/陈述拆分/反向提问）
├── testset.py        # 评测数据集 Schema + 自动生成器
├── ragas_metrics.py  # 6 大 RAGAS 指标实现
├── runner.py         # 评测执行器（检索→生成→评分全流程）
├── reporter.py       # 报告生成（JSON / Markdown / 分析）
├── metrics.py        # [兼容保留] 原有 EvalResult / RAGEvaluator
└── README.md
```

## 6 大评估指标

| 指标 | 阶段 | 需要 LLM | 需要 Ground Truth | 说明 |
|------|------|----------|-------------------|------|
| Recall@K / MRR | 检索 | ❌ | ✅ relevant_ids | 传统命中率 |
| Context Precision | 检索 | ✅ | ❌ | 检索文档是否相关（位置加权） |
| Context Recall | 检索 | ✅ | ✅ key_facts | 关键信息是否被覆盖 |
| Faithfulness | 生成 | ✅ | ❌ | 答案是否基于上下文（无幻觉） |
| Answer Relevancy | 生成 | ✅ | ❌ | 答案是否切题 |
| Answer Correctness | 生成 | ✅ | ✅ answer | 答案是否事实正确 |

## 快速开始

```python
from app.core.evaluation import (
    LLMJudge, EvalTestSet, RAGEvalRunner, quick_eval
)

# 1. 评判器
judge = LLMJudge(llm=my_llm)

# 2. 快速单条评估
metrics = await quick_eval(
    query="什么是 RRF 融合？",
    answer="RRF 是一种...",
    contexts=["RRF 全称 Reciprocal Rank Fusion..."],
    judge=judge,
)

# 3. 完整评估
testset = EvalTestSet.from_json("my_testset.json")
runner = RAGEvalRunner(
    judge=judge,
    retrieve_fn=my_retrieve_func,
    generate_fn=my_generate_func,
)
report = await runner.run_full(testset)
report.to_markdown("report.md")
```

## 测试集自动生成

```python
from app.core.evaluation import TestSetGenerator

gen = TestSetGenerator(llm=my_llm, milvus=milvus_mgr, embedding=emb)
testset = await gen.generate(sample_size=50, questions_per_chunk=2)
testset.to_json("eval_testset.json")  # 导出供人工抽检
```

## 相关文档

- RAGAS 论文：[RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)
- 检索管线：`app/core/retrieval/`
- RAG 生成：`app/core/rag/`
