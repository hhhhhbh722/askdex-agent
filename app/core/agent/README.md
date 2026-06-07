# Agent 编排

多模式 AI Agent 系统：ReAct 循环、Plan-and-Execute、反射评估。

## 模块结构

```
agent/
├── __init__.py        # 模块导出
├── orchestrator.py    # AgentOrchestrator：主入口，模式路由
├── react_agent.py     # ReActAgent：Thought → Action → Observation 循环
├── planner.py         # PlannerAgent：Plan-and-Execute 模式
├── reflection.py      # ReflectionAgent：事后质量评估与重试
└── README.md
```

## 模式说明

### ReAct 模式（react_agent.py）

```
Thought → Action → Action Input → Observation → ... → Final Answer
```

- 最大步数可配置（默认 8 步）
- 集成 ToolRegistry 调用工具
- 集成 MemoryManager 获取对话上下文
- 支持 trace 回调

### Plan-Execute 模式（planner.py）

```
Query → Plan（子任务列表）→ Execute（逐个执行）→ Replan（失败时）→ Result
```

- 适合复杂多步骤任务
- 最大重规划次数可配置（默认 2 次）
- 执行失败自动降级到 ReAct

### 反射评估（reflection.py）

```
Agent 输出 → ReflectionAgent.reflect()
  ├─ 质量打分（0-100）
  ├─ 幻觉检测
  ├─ 完整性检查
  └─ 重试/警告决策
```

## 配置项

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `react_max_steps` | 8 | ReAct 最大步数 |
| `max_replan_attempts` | 2 | Plan-Execute 重规划次数 |
| `fallback_react_on_plan_failure` | True | Plan 失败降级 ReAct |
| `enable_reflection` | True | 是否启用事后反射 |
| `reflection_min_quality` | 60 | 触发重试的最低质量分 |

## 相关文档

- 工具系统：`app/core/tools/`
- 记忆管理：`app/core/memory/`
- RAG 检索：`app/core/rag/`
