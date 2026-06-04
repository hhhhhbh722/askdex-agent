# -*- coding: utf-8 -*-
"""Query 增强：HyDE（假设性文档嵌入）+ 关键词提取。"""
from __future__ import annotations

from loguru import logger


async def enhance_query(query: str, llm=None) -> list[str]:
    """
    生成多条增强查询：
    1. 原文
    2. HyDE——让 LLM 生成假设性答案，用答案嵌入检索
    3. 关键词提取
    """
    queries = [query]

    if llm:
        try:
            hyde = await _generate_hyde(query, llm)
            if hyde and len(hyde) > 10:
                queries.append(hyde)
        except Exception as e:
            logger.debug("HyDE 生成失败: {}", e)

    # 关键词提取（简单规则 + LLM fallback）
    keywords = _extract_keywords(query)
    if keywords:
        queries.append(" ".join(keywords))

    return queries


async def _generate_hyde(query: str, llm) -> str:
    """让 LLM 生成一个假设性答案用于检索。"""
    prompt = (
        "你是一个知识检索助手。请根据以下问题，用中文写一段简短的假设性回答（50-150字）。"
        "这个回答不需要完全准确，目的是帮助搜索引擎找到相关文档。\n\n"
        f"问题：{query}\n\n假设性回答："
    )
    msgs = [{"role": "user", "content": prompt}]
    resp = await llm.acomplete(msgs, temperature=0.3)
    return resp[:300] if resp else ""


def _extract_keywords(query: str) -> list[str]:
    """简单关键词提取：保留名词短语和数字。"""
    import re
    # 去停用词
    stopwords = {"的", "是", "了", "吗", "呢", "什么", "怎么", "如何", "为什么",
                 "多少", "哪些", "哪个", "可以", "帮我", "请问", "一下", "这个"}
    tokens = re.findall(r"[一-鿿]+|[a-zA-Z0-9]+", query)
    keywords = [t for t in tokens if t.lower() not in stopwords and len(t) > 1]
    return keywords[:8]
