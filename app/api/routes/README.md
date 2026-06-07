# API 路由

FastAPI 路由处理器，挂载在 `/api/v1` 前缀下。

## 路由文件

```
routes/
├── __init__.py
├── chat.py           # POST /chat, /chat/stream, GET /metrics, /chat/memory
├── conversation.py   # CRUD /conversations
├── document.py       # 文档上传/检索/删除/评估
├── health.py         # GET /health, /health/ready, /health/full
├── kg.py             # 知识图谱 API
└── README.md
```

## 端点一览

### Chat（chat.py）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat` | 非流式 Agent 对话 |
| POST | `/chat/stream` | SSE 流式对话 |
| GET | `/metrics` | 请求/Token 统计 |
| GET | `/chat/memory` | 调试：查看会话记忆 |

### Document（document.py）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/documents/upload` | 单文件上传 |
| POST | `/documents/batch-upload` | 批量上传（后台任务） |
| GET | `/retrieve` | RAG 向量检索 |
| GET | `/evaluate` | RAG 简单自评 |
| DELETE | `/documents/{id}` | 删除文档 |

### KG（kg.py）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/kg/stats` | 知识图谱统计 |
| POST | `/kg/search` | 实体搜索 |
| POST | `/kg/query` | 图谱查询 |

## 相关文档

- Agent：`app/core/agent/`
- RAG：`app/core/rag/`
- ETL：`app/etl/`
