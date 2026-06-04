# -*- coding: utf-8 -*-
"""ToolRouter 工具路由测试。"""

from __future__ import annotations

import pytest

from app.core.tools.base import BaseTool
from app.core.tools.router import ToolRouter


class _FakeTool(BaseTool):
    """测试用虚拟工具。"""

    def __init__(self, name: str = "fake", description: str = "A test tool") -> None:
        super().__init__()
        self.name = name
        self.description = description

    async def execute(self, **kwargs):
        return "fake"


def _make_tools(*specs: tuple[str, str]) -> list[BaseTool]:
    """快捷创建工具列表：(name, description)。"""
    return [_FakeTool(name=n, description=d) for n, d in specs]


@pytest.fixture
def router() -> ToolRouter:
    return ToolRouter(max_tools=5)


class TestToolRouter:
    """工具路由器基于关键词的筛选。"""

    async def test_route_by_name_match(self, router: ToolRouter) -> None:
        tools = _make_tools(
            ("search", "Web search engine"),
            ("calculator", "Math calculator"),
        )
        result = await router.route("use search to find", tools)
        assert len(result) > 0
        assert result[0].name == "search"

    async def test_route_by_description_token(self, router: ToolRouter) -> None:
        tools = _make_tools(
            ("tool_a", "Helps with search tasks"),
            ("tool_b", "Handles database queries"),
        )
        result = await router.route("I need database help", tools)
        assert result[0].name == "tool_b"

    async def test_route_calculator_keyword(self, router: ToolRouter) -> None:
        tools = _make_tools(
            ("search", "Web search"),
            ("calculator", "Math calculations"),
        )
        result = await router.route("算一下 2+3 等于多少", tools)
        assert result[0].name == "calculator"

    async def test_route_search_keyword(self, router: ToolRouter) -> None:
        tools = _make_tools(
            ("search", "Web search engine"),
            ("calculator", "Math calculator"),
        )
        result = await router.route("查一下今天的新闻", tools)
        assert result[0].name == "search"

    async def test_route_empty_query_fallback(self, router: ToolRouter) -> None:
        tools = _make_tools(("a", "First"), ("b", "Second"))
        result = await router.route("", tools)
        # 全零分时返回前 max_tools 个
        assert len(result) <= 5

    async def test_route_respects_max_tools(self, router: ToolRouter) -> None:
        router_small = ToolRouter(max_tools=2)
        tools = _make_tools(
            ("a", "Tool A"),
            ("b", "Tool B"),
            ("c", "Tool C"),
            ("d", "Tool D"),
        )
        result = await router_small.route("tool", tools)
        assert len(result) == 2

    async def test_route_no_available_tools(self, router: ToolRouter) -> None:
        result = await router.route("anything", [])
        assert result == []
