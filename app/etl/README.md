# 文档 ETL 管线

文档解析 → 分块 → 向量化嵌入的完整提取-转换-加载流程。

## 模块结构

```
etl/
├── __init__.py
├── parser.py    # DocumentParser：PDF / TXT 解析
├── chunker.py   # DocumentChunker：固定/递归/段落分块策略
├── pipeline.py  # ETLPipeline：编排 parse → chunk → callback
└── README.md
```

## 分块策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `fixed` | 固定大小分块（chunk_size, overlap） | 通用文档 |
| `recursive` | 递归分块（按段落→句子→字符逐步切分） | 结构化文档 |
| `paragraph` | 按段落边界分块 | 报告、论文 |

## 使用示例

```python
from app.etl import ETLPipeline

pipeline = ETLPipeline(
    parser=DocumentParser(),
    chunker=DocumentChunker(strategy="recursive", chunk_size=500, overlap=50),
)

# 从文件读取
chunks = pipeline.run_bytes(pdf_bytes, filename="report.pdf")

# 带回调（如写入向量库）
chunks = pipeline.run_bytes(
    pdf_bytes,
    filename="report.pdf",
    on_chunks=lambda c: embed_chunks(c),
)
```

## 相关文档

- 向量库：`app/infrastructure/vectordb/`
- RAG 检索：`app/core/retrieval/`
- 文档 API：`app/api/routes/document.py`
