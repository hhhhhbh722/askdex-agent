# 测试

pytest 测试套件，覆盖核心模块的单元测试和集成测试。

## 测试结构

```
tests/
├── conftest.py           # 共享 fixtures + Mock 类（MockLLM, MockMilvus, etc.）
├── test_core/            # Agent 核心
│   ├── test_orchestrator.py
│   ├── test_react_agent.py
│   ├── test_planner.py
│   ├── test_reflection.py
│   └── test_circuit_breaker.py
├── test_rag/             # RAG 相关（待扩展）
├── test_etl/             # ETL 管线
│   ├── test_parser.py
│   ├── test_chunker.py
│   └── test_pipeline.py
├── test_infrastructure/  # 基础设施
│   ├── test_session.py
│   └── test_tracer.py
├── test_tools/           # 工具系统
├── test_intent/          # 意图识别
├── test_memory/          # 记忆系统
└── README.md
```

## 运行测试

```bash
# 全部测试
pytest

# 带覆盖率
pytest --cov=app --cov-report=html

# 跳过慢速测试
pytest -m "not slow"

# 仅运行特定模块
pytest tests/test_rag/
```

## Mock 工具

`conftest.py` 提供丰富的 mock 类：

- `MockLLM` — 可编程 LLM（add_response / add_conditional_response）
- `MockMilvusCollection` — 内存向量库
- `MockEmbedding` — 确定性哈希 embedding
- `MockRedisClient` — 内存 Redis
- `MockToolInvoker` — 可编程工具

## 相关文档

- 项目配置：`pyproject.toml`（pytest 配置）
- 评估模块：`app/core/evaluation/`
