# -*- coding: utf-8 -*-
"""内置网络搜索工具：Bing（国内可用）+ DuckDuckGo 降级。"""
from __future__ import annotations

import re
from typing import Any

import httpx
from loguru import logger

from app.core.tools.base import BaseTool, ToolParameter


class WebSearchTool(BaseTool):
    """多引擎网络搜索（Bing 优先，DuckDuckGo 降级）。"""

    def __init__(self, timeout: float = 15.0) -> None:
        super().__init__()
        self.name = "web_search"
        self.description = "在互联网上搜索关键词并返回摘要文本。"
        self.parameters = [
            ToolParameter(name="query", type="string", description="搜索关键词或完整问句", required=True)
        ]
        self._timeout = timeout

    async def execute(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            raise ValueError("参数 query 不能为空")

        # 依次尝试不同引擎
        for engine in [self._search_bing, self._search_duckduckgo]:
            try:
                result = await engine(query)
                if result and len(result) > 20:
                    return result
            except Exception as e:
                logger.debug("搜索引擎 {} 失败: {}", engine.__name__, e)

        return f"搜索「{query}」暂时不可用，请稍后重试。"

    async def _search_bing(self, query: str) -> str:
        """Bing 搜索（国内可用）。"""
        url = "https://www.bing.com/search"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            resp = await client.get(url, params={"q": query, "mkt": "zh-CN"}, headers=headers)
            resp.raise_for_status()
            # 提取搜索结果摘要
            snippets = re.findall(r'<p[^>]*class=["\'][^"\']*b_lineclamp[^"\']*["\'][^>]*>(.*?)</p>', resp.text, re.S)
            if not snippets:
                snippets = re.findall(r'<p[^>]*>(.*?)</p>', resp.text, re.S)
            texts = [re.sub(r'<[^>]+>', '', s).strip() for s in snippets[:5] if len(re.sub(r'<[^>]+>', '', s).strip()) > 10]
            if texts:
                result = f"搜索「{query}」的结果（Bing）：\n" + "\n".join(f"- {t}" for t in texts)
                logger.info("Bing 搜索成功 query={} snippets={}", query, len(texts))
                return result[:4000]
            return ""

    async def _search_duckduckgo(self, query: str) -> str:
        """DuckDuckGo 搜索（海外降级）。"""
        url = "https://html.duckduckgo.com/html/"
        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            resp = await client.post(url, data={"q": query})
            resp.raise_for_status()
            text = resp.text[:4000]
            logger.info("DuckDuckGo 搜索完成 query={}", query)
            return f"搜索「{query}」的原始结果片段：\n{text}"
