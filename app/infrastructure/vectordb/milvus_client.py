# -*- coding: utf-8 -*-
"""Milvus 向量库管理器：BM25 混合检索 + 中文分词。"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

try:
    from pymilvus import (
        AnnSearchRequest,
        Collection, CollectionSchema, DataType, FieldSchema, Function, FunctionType,
        WeightedRanker,
        connections, utility,
    )
except ImportError:
    Collection = Any; CollectionSchema = Any; DataType = Any; FieldSchema = Any
    Function = Any; FunctionType = Any; WeightedRanker = Any
    AnnSearchRequest = Any; connections = Any; utility = Any

_CONTENT_SPARSE = "content_sparse"


def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))


class MilvusManager:
    """Milvus 向量数据库管理器：BM25 + 密集向量混合检索。"""

    def __init__(self, host: str = "localhost", port: str = "19530",
                 alias: str = "default", **conn_kwargs: Any) -> None:
        self._host = host; self._port = port; self._alias = alias
        self._conn_kwargs = conn_kwargs; self._connected = False

    async def _ensure_connection(self) -> None:
        if self._connected: return
        await _run_sync(lambda: connections.connect(
            alias=self._alias, host=self._host, port=self._port, **self._conn_kwargs))
        self._connected = True
        logger.info("已连接 Milvus {}:{}", self._host, self._port)

    # ---- 集合创建 ----

    async def create_collection(self, name: str, dim: int) -> None:
        """创建带 BM25 的向量集合；已存在则跳过。"""
        await self._ensure_connection()

        def _create() -> None:
            if utility.has_collection(name, using=self._alias):
                logger.info("集合 [{}] 已存在，跳过创建", name)
                return
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535,
                            enable_analyzer=True, analyzer_params={"type": "chinese"}),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name=_CONTENT_SPARSE, dtype=DataType.SPARSE_FLOAT_VECTOR),
            ]
            schema = CollectionSchema(fields=fields, description=f"混合检索集合 {name}",
                                       enable_dynamic_field=False)
            col = Collection(name, schema, using=self._alias)

            # BM25 函数——Milvus 自动对 content 字段做中文分词生成稀疏向量
            bm25_fn = Function(
                name="content_bm25", function_type=FunctionType.BM25,
                input_field_names=["content"], output_field_names=[_CONTENT_SPARSE])
            schema.add_function(bm25_fn)

            # 密集向量索引
            col.create_index(field_name="embedding", index_params={
                "index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}})
            # 稀疏向量索引
            col.create_index(field_name=_CONTENT_SPARSE, index_params={
                "index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
            col.load()
            logger.info("已创建集合 [{}] dim={} (BM25 + Dense)", name, dim)

        try:
            await _run_sync(_create)
        except Exception as exc:
            logger.exception("创建集合失败: {}", exc)
            raise

    # ---- 插入 ----

    async def insert(self, collection: str, vectors: list[list[float]],
                     metadata: list[dict[str, Any]]) -> list[str]:
        await self._ensure_connection()
        ids = [str(m.get("id", f"auto_{i}")) for i, m in enumerate(metadata)]
        contents = [str(m.get("content", ""))[:65000] for m in metadata]
        sources = [str(m.get("source", ""))[:500] for m in metadata]
        indexes = [int(m.get("chunk_index", i)) for i, m in enumerate(metadata)]

        def _insert() -> None:
            col = Collection(collection, using=self._alias)
            col.insert([ids, contents, sources, indexes, vectors])
            col.flush()

        try:
            await _run_sync(_insert)
            return ids
        except Exception as exc:
            logger.exception("插入失败: {}", exc)
            raise

    # ---- 向量检索 ----

    async def search(self, collection: str, query_vector: list[float],
                     top_k: int = 10, **kw) -> list[dict[str, Any]]:
        """纯密集向量检索。"""
        await self._ensure_connection()
        def _search():
            col = Collection(collection, using=self._alias); col.load()
            res = col.search(data=[query_vector], anns_field="embedding",
                             param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                             limit=top_k, output_fields=["id", "content", "source"])
            return _parse_hits(res)
        return await _run_sync(_search)

    # ---- 混合检索（向量 + BM25） ----

    async def hybrid_search(self, collection: str, query_vector: list[float],
                            query_text: str, top_k: int = 10,
                            vector_weight: float = 0.7, bm25_weight: float = 0.3,
                            rerank_top_k: int = 50) -> list[dict[str, Any]]:
        """BM25 + Dense 混合检索 + WeightedRanker 加权融合。"""
        await self._ensure_connection()

        def _search():
            col = Collection(collection, using=self._alias); col.load()

            # 密集向量请求
            dense_req = AnnSearchRequest(
                data=[query_vector], anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=rerank_top_k)

            # BM25 稀疏请求——Milvus 自动对 query_text 做中文分词
            sparse_req = AnnSearchRequest(
                data=[query_text], anns_field=_CONTENT_SPARSE,
                param={"metric_type": "IP"}, limit=rerank_top_k)

            ranker = WeightedRanker(vector_weight, bm25_weight)
            res = col.hybrid_search(
                reqs=[dense_req, sparse_req], rerank=ranker,
                limit=top_k, output_fields=["id", "content", "source"])
            return _parse_hits(res)

        try:
            return await _run_sync(_search)
        except Exception as exc:
            logger.warning("混合检索失败，降级为纯向量检索: {}", exc)
            return await self.search(collection, query_vector, top_k)

    # ---- 删除 ----

    async def delete(self, collection: str, ids: list[str]) -> None:
        await self._ensure_connection()
        if not ids: return
        expr = "id in [" + ", ".join(f'"{i}"' for i in ids) + "]"

        def _del():
            Collection(collection, using=self._alias).delete(expr)
        await _run_sync(_del)


def _parse_hits(res) -> list[dict[str, Any]]:
    out = []
    for hits in res:
        for hit in hits:
            out.append({"id": hit.entity.get("id"), "distance": float(hit.distance),
                        "content": hit.entity.get("content", ""),
                        "source": hit.entity.get("source", "")})
    return out
