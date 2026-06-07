# 知识图谱

从文档中自动提取实体和关系，构建可查询的知识图谱。

## 模块结构

```
kg/
├── __init__.py
├── extractor.py   # 实体关系抽取
├── builder.py     # 图谱构建
├── service.py     # 检索服务（retrieve_kg_context）
├── enricher.py    # LLM 图谱增强
└── README.md
```

## 数据模型

| 表 | 说明 |
|----|------|
| `kg_entities` | 知识图谱实体 |
| `kg_relations` | 实体间关系（subject → predicate → object）|

## 相关文档

- RAG 检索：`app/core/retrieval/`
- 数据库模型：`app/infrastructure/database/models.py`
- KG API：`app/api/routes/kg.py`
