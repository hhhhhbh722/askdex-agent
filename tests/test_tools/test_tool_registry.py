# -*- coding: utf-8 -*-
"""ToolRegistry 工具注册中心测试。"""

from __future__ import annotations

import pytest

from app.core.tools.base import BaseTool
from app.core.tools.registry import ToolRegistry


class _FakeTool(BaseTool):
    """测试用虚拟工具。"""

    def __init__(self, name: str = "fake", description: str = "A test tool") -> None:
        super().__init__()
        self.name = name
        self.description = description

    async def execute(self, **kwargs):
        return "fake_result"


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


class TestToolRegistry:
    """工具注册中心基本操作。"""

    def test_register_and_get(self, registry: ToolRegistry) -> None:
        tool = _FakeTool(name="test_tool", description="For testing")
        registry.register(tool)

        retrieved = registry.get_tool("test_tool")
        assert retrieved is tool
        assert retrieved.name == "test_tool"

    def test_register_duplicate_overwrites(self, registry: ToolRegistry) -> None:
        tool1 = _FakeTool(name="dup", description="First")
        tool2 = _FakeTool(name="dup", description="Second")

        registry.register(tool1)
        registry.register(tool2)

        assert registry.get_tool("dup") is tool2

    def test_get_nonexistent_raises_keyerror(self, registry: ToolRegistry) -> None:
        with pytest.raises(KeyError, match="未注册的工具"):
            registry.get_tool("nonexistent")

    def test_get_all_tools(self, registry: ToolRegistry) -> None:
        t1 = _FakeTool("t1")
        t2 = _FakeTool("t2")
        registry.register(t1)
        registry.register(t2)

        all_tools = registry.get_all_tools()
        assert len(all_tools) == 2
        assert t1 in all_tools
        assert t2 in all_tools

    def test_get_tools_description(self, registry: ToolRegistry) -> None:
        t1 = _FakeTool("search", "Search the web")
        t1.parameters = []  # type: ignore[assignment]
        registry.register(t1)

        desc = registry.get_tools_description()
        assert "search" in desc
        assert "Search the web" in desc

    def test_empty_registry(self, registry: ToolRegistry) -> None:
        assert registry.get_all_tools() == []
        desc = registry.get_tools_description()
        assert "无可用工具" in desc
