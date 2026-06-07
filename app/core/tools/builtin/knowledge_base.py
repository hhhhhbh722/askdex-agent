# -*- coding: utf-8 -*-
"""Knowledge-base search tool for the agent runtime."""
from __future__ import annotations

from typing import Awaitable, Callable

from app.core.tools.base import ToolParameter

SearchFn = Callable[..., Awaitable[list[dict]]]


class KnowledgeBaseTool:
    """Agent tool: query uploaded knowledge-base documents."""

    name = "knowledge_base"
    description = "查询知识库，搜索已上传的文档内容。输入 query 参数指定搜索关键词。"

    def __init__(self, search_fn: SearchFn):
        self._search_fn = search_fn
        self.parameters = [
            ToolParameter(name="query", type="string", description="要在知识库中检索的问题或关键词", required=True)
        ]

    def schema_parameters(self):
        properties = {p.name: {"type": p.type, "description": p.description} for p in self.parameters}
        required = [p.name for p in self.parameters if p.required]
        return {"type": "object", "properties": properties, "required": required}

    async def execute(self, **kwargs):
        query = str(kwargs.get("query", "") or kwargs.get("expression", ""))
        if not query:
            return "请提供 query 参数"

        results = await self._search_fn(query, top_k=5)
        if not results:
            return "知识库中未找到相关内容"

        lines = []
        for i, item in enumerate(results, 1):
            score = float(item.get("score") or 0)
            lines.append(f"[{i}] (相关度={score:.2f})\n{str(item.get('content') or '')[:300]}")
        return "\n\n".join(lines)
