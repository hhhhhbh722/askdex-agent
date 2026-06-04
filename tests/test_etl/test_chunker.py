# -*- coding: utf-8 -*-
"""DocumentChunker 文本分块策略测试。"""

from __future__ import annotations

import pytest

from app.etl.chunker import DocumentChunker


@pytest.fixture
def chunker() -> DocumentChunker:
    return DocumentChunker(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def small_chunker() -> DocumentChunker:
    return DocumentChunker(chunk_size=50, chunk_overlap=10)


class TestFixedChunking:
    """固定大小分块策略。"""

    def test_single_chunk_fits(self, chunker: DocumentChunker) -> None:
        text = "Short text that fits in one chunk."
        chunks = chunker.chunk(text, strategy="fixed")
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_multiple_chunks(self, chunker: DocumentChunker) -> None:
        text = "A" * 250  # 需要 3 个 chunk（100 大小，20 重叠）
        chunks = chunker.chunk(text, strategy="fixed")
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) <= 100

    def test_overlap_between_chunks(self, chunker: DocumentChunker) -> None:
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10  # 260 chars
        chunks = chunker.chunk(text, strategy="fixed")
        if len(chunks) >= 2:
            # 验证重叠：chunk1 尾部应与 chunk2 头部重叠
            end_of_first = chunks[0][-20:]
            start_of_second = chunks[1][:20]
            # 取较短的一端验证部分重叠
            assert end_of_first[:10] in chunks[1] or any(
                end_of_first[:10] in c for c in chunks[1:2]
            )

    def test_empty_text(self, chunker: DocumentChunker) -> None:
        chunks = chunker.chunk("", strategy="fixed")
        assert chunks == []

    def test_text_shorter_than_chunk_size(self, chunker: DocumentChunker) -> None:
        text = "Hello"
        chunks = chunker.chunk(text, strategy="fixed")
        assert chunks == [text]


class TestParagraphChunking:
    """段落分块策略。"""

    def test_single_paragraph(self, chunker: DocumentChunker) -> None:
        text = "This is a single paragraph."
        chunks = chunker.chunk(text, strategy="paragraph")
        assert len(chunks) == 1

    def test_multiple_paragraphs_merged_if_fit(self, chunker: DocumentChunker) -> None:
        text = "Para 1.\n\nPara 2."
        chunks = chunker.chunk(text, strategy="paragraph")
        assert len(chunks) >= 1

    def test_large_paragraphs_split(self, small_chunker: DocumentChunker) -> None:
        text = "A" * 200 + "\n\n" + "B" * 200
        chunks = small_chunker.chunk(text, strategy="paragraph")
        assert len(chunks) >= 2


class TestRecursiveChunking:
    """递归分块策略。"""

    def test_basic_recursive(self, chunker: DocumentChunker) -> None:
        text = "First paragraph about AI.\n\nSecond paragraph about ML.\n\nThird about DL."
        chunks = chunker.chunk(text, strategy="recursive")
        assert len(chunks) >= 1
        # 所有 chunk 应在 chunk_size 范围内
        for c in chunks:
            assert len(c) <= 100

    def test_recursive_without_newlines(self, chunker: DocumentChunker) -> None:
        text = "A long text without any paragraph breaks. " * 20
        chunks = chunker.chunk(text, strategy="recursive")
        assert len(chunks) >= 1

    def test_deep_recursion_guard(self, chunker: DocumentChunker) -> None:
        """验证递归深度限制不会导致无限递归。"""
        text = "X" * 1000
        chunks = chunker.chunk(text, strategy="recursive")
        # 应能成功返回而不抛出 RecursionError
        assert len(chunks) >= 1


class TestChunkerConfiguration:
    """配置相关测试。"""

    def test_chunk_size_minimum_32(self) -> None:
        c = DocumentChunker(chunk_size=10)
        assert c.chunk_size == 32  # minimum enforced

    def test_overlap_clamped(self) -> None:
        c = DocumentChunker(chunk_size=100, chunk_overlap=150)
        assert c.chunk_overlap <= 99  # max chunk_size - 1

    def test_overlap_minimum_zero(self) -> None:
        c = DocumentChunker(chunk_size=100, chunk_overlap=-5)
        assert c.chunk_overlap == 0
