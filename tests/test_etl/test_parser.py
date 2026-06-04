# -*- coding: utf-8 -*-
"""DocumentParser 文档解析测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.etl.parser import DocumentParser


@pytest.fixture
def parser() -> DocumentParser:
    return DocumentParser(max_chars=5000)


class TestTextParsing:
    """纯文本文件解析测试。"""

    def test_parse_txt_file(self, parser: DocumentParser, tmp_path: Path) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello, world!", encoding="utf-8")

        result = parser.parse_file(file_path, mime_type="text/plain")
        assert result.text == "Hello, world!"
        assert result.mime_type == "text/plain"

    def test_parse_txt_bytes(self, parser: DocumentParser) -> None:
        data = b"Hello from bytes!"
        result = parser.parse_bytes(data, filename="test.txt", mime_type="text/plain")
        assert result.text == "Hello from bytes!"
        assert result.mime_type == "text/plain"

    def test_parse_txt_chinese(self, parser: DocumentParser, tmp_path: Path) -> None:
        file_path = tmp_path / "chinese.txt"
        file_path.write_text("你好世界！人工智能代理测试。", encoding="utf-8")

        result = parser.parse_file(file_path, mime_type="text/plain")
        assert "你好世界" in result.text
        assert "人工智能" in result.text

    def test_parse_max_chars_truncation(self, tmp_path: Path) -> None:
        parser_small = DocumentParser(max_chars=10)
        file_path = tmp_path / "long.txt"
        file_path.write_text("A" * 100, encoding="utf-8")

        result = parser_small.parse_file(file_path, mime_type="text/plain")
        assert len(result.text) <= 10


class TestUnsupportedTypes:
    """不支持的文件类型处理。"""

    def test_parse_unsupported_file_type(self, parser: DocumentParser, tmp_path: Path) -> None:
        file_path = tmp_path / "test.xyz"
        file_path.write_text("content", encoding="utf-8")

        result = parser.parse_file(file_path)
        assert result.text == "" or result.meta.get("warning") == "unsupported"

    def test_parse_unknown_bytes_type(self, parser: DocumentParser) -> None:
        result = parser.parse_bytes(b"content", filename="unknown.xyz", mime_type=None)
        # 应用不应崩溃
        assert result is not None
