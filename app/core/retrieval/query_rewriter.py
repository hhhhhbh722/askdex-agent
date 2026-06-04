# -*- coding: utf-8 -*-
"""Query Rewrite：多轮对话上下文改写（参考 gov_rag 设计）。"""
from __future__ import annotations

from loguru import logger

_REWRITE_PROMPT = """你是一个问题改写助手。根据对话历史，将用户最新提问改写成完整、独立的问题。

规则：
1. 如果最新提问本身完整清晰，直接原样返回。
2. 如果依赖上下文（如"那这个呢"、"例外情况呢"、"还有吗"），结合历史补全指代词和缺失信息。
3. 只输出改写后的问题，不要任何解释。

对话历史：
{history}

用户最新提问：{query}

改写后的完整问题："""


async def rewrite_query(query: str, history: list[dict], llm) -> str:
    """
    多轮对话 Query 改写。
    history: [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
    """
    # 第一轮不需要改写
    user_msgs = [m for m in history if m.get("role") == "user"]
    if len(user_msgs) <= 1:
        return query

    # 只取最近 4 轮
    recent = history[-8:]
    lines = []
    for m in recent:
        if m.get("role") == "user" and m.get("content") == query:
            continue
        label = "用户" if m.get("role") == "user" else "助手"
        lines.append(f"{label}：{m.get('content', '')}")
    hist_str = "\n".join(lines)
    if not hist_str.strip():
        return query

    prompt = _REWRITE_PROMPT.format(history=hist_str, query=query)
    try:
        resp = await llm.acomplete([{"role": "user", "content": prompt}], temperature=0.0)
        rewritten = resp.strip() if resp else ""
        if rewritten and rewritten != query:
            logger.info("Query Rewrite: '{}' → '{}'", query[:40], rewritten[:80])
            return rewritten
    except Exception as e:
        logger.warning("Query Rewrite 失败，降级为原始查询: {}", e)
    return query
