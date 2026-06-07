# 基础设施

外部服务封装：LLM、数据库、向量库、缓存、链路追踪。

## 模块结构

```
infrastructure/
├── llm/           # LLM 调用封装（ModelRouter, CircuitBreaker）
├── database/      # SQLAlchemy 异步引擎 + ORM 模型
├── vectordb/      # Milvus 混合检索（Dense + BM25 Sparse）
├── cache/         # Redis 缓存封装
├── trace/         # 链路追踪（InMemoryTracer）
└── README.md
```

## 各模块说明

### LLM（infrastructure/llm/）

- **ModelRouter**：多模型路由（优先级、权重、熔断）
- **CircuitBreaker**：熔断器（CLOSED → OPEN → HALF_OPEN 状态机）
- **types**：LLM 相关类型定义

### 数据库（infrastructure/database/）

- **models.py**：SQLAlchemy ORM（Conversation, Message, Document, DocumentChunk, KGEntity, KGRelation）
- **session.py**：异步 session 工厂

### 向量库（infrastructure/vectordb/）

- **MilvusManager**：
  - Dense 向量检索（COSINE 距离）
  - BM25 稀疏向量混合检索（WeightedRanker）
  - 中文分词器（CJK bigrams + 英文 token）
  - SPARSE_INVERTED_INDEX + IVF_FLAT 双索引

### 缓存（infrastructure/cache/）

- Redis 封装，用于短期记忆存储

### 链路追踪（infrastructure/trace/）

- **Tracer**：内存 span tree
- 支持检索 trace、Agent trace

## 相关文档

- Agent：`app/core/agent/`
- RAG 检索：`app/core/retrieval/`
- 记忆管理：`app/core/memory/`
