# 企业级 AI Agent 架构

## RAG 检索流水线

```
用户提问: "小夜的蛋多重？"
    │
    ▼
┌─ 1. Query 增强 ───────────────────────────────────┐
│  HyDE: LLM 生成假设性答案用于检索                   │
│  "小夜的蛋是一种小型宠物蛋，重约0.5-1克"              │
│  关键词: "小夜" "蛋" "重量"                          │
│  → 3 条增强查询                                     │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌─ 2. 混合检索 (Milvus Hybrid Search) ─────────────┐
│  ┌─ 密集向量 (COSINE, nprobe=16)                  │
│  ├─ BM25 稀疏向量 (中文分词, FunctionType.BM25)    │
│  └─ WeightedRanker (0.7 Dense + 0.3 BM25)        │
│  → Recall Top-50                                  │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌─ 3. Reranker 精排 ───────────────────────────────┐
│  DashScope gte-rerank API                         │
│  → Cross-Encoder 逐对打分                          │
│  → 精选 Top-5                                      │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌─ 4. Agent 推理 (ReAct) ──────────────────────────┐
│  Thought: 知识库检索到相关数据                      │
│  Action: 综合分析                                  │
│  Final Answer: 基于引用内容给出答案                  │
└──────────────────────────────────────────────────┘
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Vue 3 + Vite + Ant Design Vue 4 |
| API | FastAPI + SSE 流式 |
| Agent | ReAct + 工具调用（calculator/web_search/knowledge_base） |
| LLM | DeepSeek v4-pro (API) |
| Embedding | DashScope text-embedding-v4 (API, 1024维) |
| Reranker | DashScope gte-rerank (API) |
| 向量库 | Milvus Standalone (BM25 + Dense Hybrid) |
| 缓存 | Redis (会话窗口) |
| 数据库 | PostgreSQL (文档元数据 + 对话历史) |

## 数据流

```
文档上传 → ETL 分块 → Embedding API → Milvus (content + vector)
对话请求 → Query 增强 → Hybrid Search → Reranker → Agent → 回答
```
