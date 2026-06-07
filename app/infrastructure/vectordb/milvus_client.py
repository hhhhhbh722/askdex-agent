# -*- coding: utf-8 -*-
"""Milvus 向量库管理器：密集向量 + 手动稀疏关键词向量混合检索。"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections import Counter
from typing import Any

from loguru import logger

try:
    from pymilvus import (
        AnnSearchRequest,
        Collection, CollectionSchema, DataType, FieldSchema,
        WeightedRanker,
        connections, utility,
    )
except ImportError:
    Collection = Any; CollectionSchema = Any; DataType = Any; FieldSchema = Any
    WeightedRanker = Any
    AnnSearchRequest = Any; connections = Any; utility = Any

_CONTENT_SPARSE = "content_sparse"
_HASH_BUCKETS = 2_000_000_000


def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))


def _tokenize(text: str) -> list[str]:
    """轻量中英文切词：英文按词，中文按单字+相邻双字。"""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)
    cjk_chars = [t for t in tokens if "\u4e00" <= t <= "\u9fff"]
    bigrams = [a + b for a, b in zip(cjk_chars, cjk_chars[1:])]
    return tokens + bigrams


def _term_id(term: str) -> int:
    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % _HASH_BUCKETS


def _sparse_vector(text: str) -> dict[int, float]:
    """生成 BM25 风格的稀疏关键词向量，供 Milvus sparse IP 检索使用。"""
    counts = Counter(_tokenize(text))
    if not counts:
        return {}
    total = sum(counts.values()) or 1
    out: dict[int, float] = {}
    for term, tf in counts.items():
        # BM25 的 TF 饱和项近似；IDF 由查询和语料变化决定，这里用稳定的
        # log-TF 权重保留关键词召回能力，避免依赖 Milvus 内置 Function。
        weight = (tf * 2.5) / (tf + 1.5) * math.log1p(total / tf)
        tid = _term_id(term)
        out[tid] = out.get(tid, 0.0) + float(weight)
    return out


class MilvusManager:
    """Milvus 向量数据库管理器：稀疏关键词 + 密集向量混合检索。"""

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
        """创建向量集合；已存在则跳过。"""
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
                FieldSchema(name="group", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="parent_group", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="child_group", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name=_CONTENT_SPARSE, dtype=DataType.SPARSE_FLOAT_VECTOR),
            ]
            schema = CollectionSchema(fields=fields, description=f"混合检索集合 {name}",
                                       enable_dynamic_field=False)
            col = Collection(name, schema, using=self._alias)

            # 密集向量索引
            col.create_index(field_name="embedding", index_params={
                "index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}})
            # 手动稀疏关键词向量索引
            col.create_index(field_name=_CONTENT_SPARSE, index_params={
                "index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP", "params": {}})
            col.load()
            logger.info("已创建集合 [{}] dim={} (Dense + Sparse)", name, dim)

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
        groups = [str(m.get("group", ""))[:500] for m in metadata]
        parent_groups = [str(m.get("parent_group", ""))[:250] for m in metadata]
        child_groups = [str(m.get("child_group", ""))[:250] for m in metadata]
        indexes = [int(m.get("chunk_index", i)) for i, m in enumerate(metadata)]
        sparse_vectors = [_sparse_vector(content) for content in contents]

        def _insert() -> None:
            col = Collection(collection, using=self._alias)
            col.insert([
                ids, contents, sources, groups, parent_groups, child_groups,
                indexes, vectors, sparse_vectors,
            ])
            col.flush()

        try:
            await _run_sync(_insert)
            return ids
        except Exception as exc:
            logger.exception("插入失败: {}", exc)
            raise

    # ---- 向量检索 ----

    async def search(self, collection: str, query_vector: list[float],
                     top_k: int = 10, expr: str | None = None, **kw) -> list[dict[str, Any]]:
        """纯密集向量检索。"""
        await self._ensure_connection()
        def _search():
            col = Collection(collection, using=self._alias); col.load()
            res = col.search(data=[query_vector], anns_field="embedding",
                             param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                             limit=top_k, expr=expr,
                             output_fields=["id", "content", "source", "group", "parent_group", "child_group"])
            return _parse_hits(res)
        return await _run_sync(_search)

    # ---- 混合检索（向量 + BM25） ----

    async def hybrid_search(self, collection: str, query_vector: list[float],
                            query_text: str, top_k: int = 10,
                            vector_weight: float = 0.7, bm25_weight: float = 0.3,
                            rerank_top_k: int = 50, expr: str | None = None) -> list[dict[str, Any]]:
        """Sparse keyword + Dense 混合检索 + WeightedRanker 加权融合。"""
        await self._ensure_connection()

        def _search():
            col = Collection(collection, using=self._alias); col.load()

            # 密集向量请求（pymilvus 3.0: expr 需在 AnnSearchRequest 上设置）
            dense_req = AnnSearchRequest(
                data=[query_vector], anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=rerank_top_k, expr=expr)

            # 稀疏关键词请求
            sparse_query = _sparse_vector(query_text)
            if not sparse_query:
                return _parse_hits(col.search(
                    data=[query_vector], anns_field="embedding",
                    param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                    limit=top_k, expr=expr,
                    output_fields=["id", "content", "source", "group", "parent_group", "child_group"]))
            sparse_req = AnnSearchRequest(
                data=[sparse_query], anns_field=_CONTENT_SPARSE,
                param={"metric_type": "IP"}, limit=rerank_top_k, expr=expr)

            ranker = WeightedRanker(vector_weight, bm25_weight)
            res = col.hybrid_search(
                reqs=[dense_req, sparse_req], rerank=ranker,
                limit=top_k,
                output_fields=["id", "content", "source", "group", "parent_group", "child_group"])
            return _parse_hits(res)

        try:
            return await _run_sync(_search)
        except Exception as exc:
            logger.warning("混合检索失败，降级为纯向量检索: {}", exc)
            return await self.search(collection, query_vector, top_k, expr=expr)

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
                        "source": hit.entity.get("source", ""),
                        "group": hit.entity.get("group", ""),
                        "parent_group": hit.entity.get("parent_group", ""),
                        "child_group": hit.entity.get("child_group", "")})
    return out
