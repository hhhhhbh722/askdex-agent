# 意图识别

用户查询意图分类与槽位提取，用于 Agent 模式选择。

## 模块结构

```
intent/
├── __init__.py
├── recognizer.py   # IntentRecognizer：意图识别 + 槽位填充
└── README.md
```

## 意图类型

- `rag_query` — 知识库检索查询
- `calculation` — 数学计算
- `web_search` — 需要联网搜索
- `general_chat` — 一般对话

## 相关文档

- Agent 编排：`app/core/agent/`
- 工具系统：`app/core/tools/`
