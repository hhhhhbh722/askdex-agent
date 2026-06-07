# AskDex Agent

精灵知识库 RAG + Agent 问答系统。Ask + Dex + Agent：对图鉴提问的智能体。

基于 FastAPI + Milvus + DeepSeek，支持文档上传、混合检索、ReAct Agent 推理和 RAGAS 评估。

## 技术栈

| 层级 | 技术 |
|------|------|
| 框架 | FastAPI + SSE 流式 |
| Agent | ReAct / Plan-Execute / Reflection |
| 检索 | Dense + BM25 混合检索 + HyDE + RRF 融合 + Reranker 精排 |
| LLM | DeepSeek v4-pro |
| Embedding | DashScope text-embedding-v4 (1024维) |
| Reranker | DashScope gte-rerank |
| 向量库 | Milvus Standalone |
| 数据库 | PostgreSQL + Redis |
| 评估 | RAGAS (Faithfulness / Context Precision / Answer Relevancy 等) |
| 前端 | Vue 3 + Vite |

## 快速开始

```bash
# 安装依赖
pip install -e .

# 启动服务（需要先启动 docker-compose 中的 Milvus / Redis / PostgreSQL）
uvicorn app.main:app --reload

# 或一键启动
docker-compose up -d
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/chat` | Agent 对话 |
| POST | `/api/v1/chat/stream` | SSE 流式对话 |
| POST | `/api/v1/documents/upload` | 上传文档 |
| GET | `/api/v1/retrieve` | 知识库检索 |
| POST | `/api/v1/evaluate/ragas` | RAGAS 评估 |
| POST | `/api/v1/evaluate/testset/generate` | 自动生成测试集 |

## 项目结构

```
app/
├── api/routes/        # API 路由
├── core/
│   ├── agent/         # ReAct / Plan-Execute / Reflection
│   ├── retrieval/     # 检索管线 (HyDE / Hybrid / RRF / Reranker)
│   ├── rag/           # RAG 生成器 + 多路检索器
│   ├── evaluation/    # RAGAS 评估体系
│   ├── memory/        # 短期记忆 (Redis) + 长期记忆 (Milvus)
│   ├── kg/            # 知识图谱抽取与检索
│   └── tools/         # Agent 工具 (计算器 / 搜索 / 知识库)
├── etl/               # 文档解析 → 分块 → 向量化
├── infrastructure/    # Milvus / PostgreSQL / Redis / LLM
└── models/            # Pydantic Schema
```
