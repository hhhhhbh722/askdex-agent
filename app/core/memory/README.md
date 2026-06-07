# 记忆管理

对话记忆系统：短期记忆（Redis 滑动窗口）+ 长期记忆（Milvus 语义检索）。

## 模块结构

```
memory/
├── __init__.py     # 模块导出
├── short_term.py   # ShortTermMemory：Redis 滑动窗口对话记忆
├── long_term.py    # LongTermMemory：Milvus 持久化语义记忆
├── manager.py      # MemoryManager：协调短期+长期记忆
└── README.md
```

## 组件说明

### ShortTermMemory（short_term.py）

- **存储**：Redis List（RPUSH / LRANGE）
- **窗口大小**：默认 20 条消息
- **Token 限制**：超过限制自动触发 LLM 压缩摘要
- **TTL**：默认 7 天，过期自动清理

### LongTermMemory（long_term.py）

- **存储**：Milvus 向量集合（语义搜索）
- **写入**：embedding + insert
- **召回**：按 session_id 过滤 + 语义相似度检索
- **遗忘**：按 ID 删除

### MemoryManager（manager.py）

协调短期和长期记忆，为 Agent 提供统一的上下文接口。

```python
manager = MemoryManager(short_term=stm, long_term=ltm)

# 获取完整上下文
context = await manager.get_context(session_id, query)
# context.short_term_messages  — 近期对话
# context.long_term_items      — 相关长期记忆

# 追加一轮对话
await manager.append_turn(session_id, user_msg, assistant_msg)
```

## 相关文档

- Agent 编排：`app/core/agent/`
- 向量库：`app/infrastructure/vectordb/`
- 缓存：`app/infrastructure/cache/`
