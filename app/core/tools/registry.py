# -*- coding: utf-8 -*-
"""工具注册中心：集中管理可用工具并生成 Prompt 描述。"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Iterable, List

from loguru import logger

from app.core.tools.base import BaseTool


class ToolRegistry:
    """工具注册中心：管理所有可用工具。"""

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._timeout_seconds = timeout_seconds

    def register(self, tool: BaseTool) -> None:
        """注册工具；同名覆盖并记录日志。"""
        if tool.name in self._tools:
            logger.warning("工具 [{}] 已存在，将被覆盖", tool.name)
        self._tools[tool.name] = tool
        logger.info("已注册工具: {}", tool.name)

    def get_tool(self, name: str) -> BaseTool:
        """按名称获取工具。"""
        if name not in self._tools:
            raise KeyError(f"未注册的工具: {name}")
        return self._tools[name]

    def get_all_tools(self) -> List[BaseTool]:
        """返回全部工具列表。"""
        return list(self._tools.values())

    def list_tool_names(self) -> list[str]:
        """供 Agent 编排器使用。"""
        return list(self._tools.keys())

    async def invoke(self, name: str, arguments: dict) -> str:
        """供 Agent 编排器调用工具。"""
        tool = self.get_tool(name)
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                tool.execute(**arguments),
                timeout=self._timeout_seconds,
            )
            duration_ms = int((time.perf_counter() - started) * 1000)
            logger.info("工具调用成功 tool={} duration_ms={}", name, duration_ms)
            return str(result)
        except asyncio.TimeoutError as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            logger.warning("工具调用超时 tool={} duration_ms={}", name, duration_ms)
            raise RuntimeError(f"工具 {name} 调用超时（{duration_ms}ms）") from exc
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            logger.warning("工具调用失败 tool={} duration_ms={} error={}", name, duration_ms, exc)
            raise RuntimeError(f"工具 {name} 调用失败（{duration_ms}ms）: {exc}") from exc

    def get_tools_description(self, names: Iterable[str] | None = None) -> str:
        """生成所有工具的自然语言描述（用于 System Prompt）。"""
        lines: list[str] = []
        selected = list(names) if names is not None else list(self._tools.keys())
        for name in selected:
            tool = self._tools.get(name)
            if not tool:
                continue
            schema = json.dumps(tool.schema_parameters(), ensure_ascii=False)
            lines.append(
                f"- {tool.name}\n"
                f"  description: {tool.description}\n"
                f"  parameters: {schema}"
            )
        return "\n".join(lines) if lines else "（当前无可用工具）"
