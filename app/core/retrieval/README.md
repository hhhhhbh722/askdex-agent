# 检索管线

多策略检索流水线：查询改写 → 多查询生成(HyDE) → 混合检索(向量+BM25) → RRF 融合 → Reranker 精排。

## 模块结构

```
retrieval/
├── __init__.py         # 模块导出（retrieval_pipeline）
├── pipeline.py         # 检索主流程编排
├── query_rewriter.py   # 多轮对话查询改写
├── query_enhancer.py   # 多查询生成（HyDE + 关键词提取）
├── reranker.py         # DashScope gte-rerank 重排序
└── README.md
```

## 检索流程

```
用户 Query
  │
  ├─→ [Query Rewrite] 多轮对话上下文改写（若有历史）
  │
  ├─→ [Multi-Query] 原始 query + 改写 query + HyDE 假设文档
  │
  ├─→ [Hybrid Search] Milvus Dense(向量) + Sparse(BM25) 混合检索
  │     └─ WeightedRanker(0.7 dense, 0.3 sparse)
  │
  ├─→ [RRF Fusion] 多路检索结果 Reciprocal Rank Fusion
  │
  └─→ [Reranker] DashScope gte-rerank 交叉编码器精排
        └─ Score 归一化（distance → 0-1）
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `RRF_K` | 60 | RRF 平滑常数 |
| `top_k` | 5 | 最终返回结果数 |
| `hybrid` | True | 是否启用 BM25 混合检索 |
| `enable_hyde` | False | 是否启用 HyDE 假设文档生成 |

## 相关文档

- 向量库：`app/infrastructure/vectordb/`
- RAG 评估：`app/core/evaluation/`
- 重排序：`app/core/retrieval/reranker.py`（DashScope API）
