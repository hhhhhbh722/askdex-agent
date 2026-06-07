# 工具系统

Agent 可调用的工具框架：注册、路由、内置工具。

## 模块结构

```
tools/
├── __init__.py
├── base.py        # ToolParameter, BaseTool 抽象类
├── registry.py    # ToolRegistry：工具注册/查找/超时控制
├── router.py      # 工具路由（按 Intent 分发）
├── builtin/       # 内置工具集
│   ├── calculator.py   # 数学计算工具
│   ├── search.py       # Web 搜索工具
│   └── README.md
└── README.md
```

## 工具接口

```python
class BaseTool:
    name: str
    description: str
    parameters: list[ToolParameter]

    async def execute(self, **kwargs) -> str: ...
    def schema_parameters(self) -> dict: ...  # → OpenAI function schema
```

## 内置工具

| 工具 | 名称 | 功能 |
|------|------|------|
| `CalculatorTool` | `calculator` | 数学表达式求值 |
| `WebSearchTool` | `web_search` | Web 搜索（DuckDuckGo） |
| `_KBTool` | `knowledge_base` | 知识库检索（在 wiring.py 中注册） |

## 注册与使用

```python
registry = ToolRegistry(timeout_seconds=90.0)
registry.register(CalculatorTool())
registry.register(WebSearchTool())

# Agent 中获取工具描述
schemas = [t.schema_parameters() for t in registry.get_all()]
```

## 相关文档

- Agent 编排：`app/core/agent/`
- RAG 检索：`app/core/rag/`
