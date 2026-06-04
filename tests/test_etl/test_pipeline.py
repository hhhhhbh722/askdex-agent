# -*- coding: utf-8 -*-
"""ETLPipeline 流水线测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.etl.chunker import DocumentChunker
from app.etl.parser import DocumentParser, ParsedDocument
from app.etl.pipeline import ETLPipeline


@pytest.fixture
def mock_parser() -> DocumentParser:
    p = MagicMock(spec=DocumentParser)
    p.parse_bytes.return_value = ParsedDocument(
        text="Test document text for chunking",
        mime_type="text/plain",
        meta={"filename": "test.txt"},
    )
    return p


@pytest.fixture
def mock_chunker() -> DocumentChunker:
    c = MagicMock(spec=DocumentChunker)
    c.chunk.return_value = ["chunk1: Test document", "chunk2: text for chunking"]
    return c


@pytest.fixture
def pipeline(mock_parser: DocumentParser, mock_chunker: DocumentChunker) -> ETLPipeline:
    return ETLPipeline(parser=mock_parser, chunker=mock_chunker)


class TestETLPipeline:
    """ETL 流水线测试。"""

    async def test_run_bytes_success(self, pipeline: ETLPipeline) -> None:
        result = await pipeline.run_bytes(b"test data", "test.txt", "text/plain")
        assert result.meta is not None
        assert result.parsed is not None
        assert len(result.chunks) == 2

    async def test_run_bytes_with_on_chunks(self, pipeline: ETLPipeline) -> None:
        chunks_collected: list[str] = []

        async def on_chunks(chunks: list[str], parsed: ParsedDocument) -> None:
            chunks_collected.extend(chunks)

        await pipeline.run_bytes(
            b"test", "test.txt", "text/plain", on_chunks=on_chunks
        )
        assert len(chunks_collected) == 2

    async def test_parser_failure_propagates(self, mock_parser: DocumentParser) -> None:
        mock_parser.parse_bytes.side_effect = RuntimeError("Parse error")
        pipeline = ETLPipeline(parser=mock_parser, chunker=DocumentChunker())

        with pytest.raises(RuntimeError, match="Parse error"):
            await pipeline.run_bytes(b"test", "test.txt", "text/plain")

    async def test_default_pipeline_works(self) -> None:
        """使用默认 parser 和 chunker 的流水线应能处理简单文本。"""
        pipeline = ETLPipeline()
        result = await pipeline.run_bytes(b"Hello world", "test.txt", "text/plain")
        assert result is not None
        assert len(result.chunks) >= 1
