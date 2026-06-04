# -*- coding: utf-8 -*-
"""数据库 session URL 标准化函数测试。"""

from __future__ import annotations

from app.infrastructure.database.session import normalize_async_database_url


class TestNormalizeDatabaseURL:
    """URL 标准化转换测试。"""

    def test_already_asyncpg(self) -> None:
        url = "postgresql+asyncpg://user:pass@localhost/db"
        assert normalize_async_database_url(url) == url

    def test_psycopg2_to_asyncpg(self) -> None:
        url = "postgresql+psycopg2://user:pass@localhost/db"
        expected = "postgresql+asyncpg://user:pass@localhost/db"
        assert normalize_async_database_url(url) == expected

    def test_postgres_short_to_asyncpg(self) -> None:
        url = "postgres://user:pass@localhost/db"
        expected = "postgresql+asyncpg://user:pass@localhost/db"
        assert normalize_async_database_url(url) == expected

    def test_plain_postgresql_to_asyncpg(self) -> None:
        url = "postgresql://user:pass@localhost/db"
        expected = "postgresql+asyncpg://user:pass@localhost/db"
        assert normalize_async_database_url(url) == expected
