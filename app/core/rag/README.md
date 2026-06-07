# RAG 子系统

检索增强生成（RAG）核心组件：多路检索器、重排序器、答案生成器。

## 模块结构

```
rag/
├── __init__.py     # 模块导出
├── retriever.py    # MultiRetriever：向量/关键词/混合检索 + RRF 融合
├── reranker.py     # CrossEncoder 重排序（sentence-transformers 本地模型）
├── generator.py    # RAGGenerator：基于上下文 + 历史生成答案 + 引用标注
└── README.md
```

## 组件说明

### MultiRetriever（retriever.py）

支持三种检索模式：

- **Vector Search**：基于 embedding 的语义检索
- **Keyword Search**：基于内存 BM25 索引的关键词检索
- **Hybrid Search**：向量 + BM25 并行检索 + RRF 融合

```python
retriever = MultiRetriever(embedding=emb, milvus=milvus)
results = await retriever.hybrid_search(query, top_k=5)
```

### Reranker（reranker.py）

基于 `sentence-transformers` 的本地交叉编码器重排序。

> **注意**：实际生产环境中，检索管线使用的是 `app/core/retrieval/reranker.py`（DashScope API 版本）。此模块为本地离线版本。

### RAGGenerator（generator.py）

基于检索上下文和对话历史生成答案，支持 `[n]` 格式的引用标注。

```python
gen = RAGGenerator(llm=my_llm)
response = await gen.generate(
    query="什么是RRF？",
    contexts=[RetrievalResult(...)],
    chat_history=[],
)
# response.answer    — 带 [1][2] 引用的答案正文
# response.citations — 解析后的 Citation 列表
```

## 相关文档

- 检索管线：`app/core/retrieval/`
- Agent 编排：`app/core/agent/`
- 评估：`app/core/evaluation/`
