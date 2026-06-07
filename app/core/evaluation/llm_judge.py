# -*- coding: utf-8 -*-
"""
LLM 评判器（Judge）。

为 RAGAS 评估指标提供统一的 LLM 调用封装，包括：
- 数值打分（score）：让 LLM 对某个维度给出 0-1 分数
- 分类判定（classify）：让 LLM 从候选标签中选择最匹配的
- 陈述拆分（extract_claims）：将一段文本拆分为原子陈述列表
- 反向问题生成（generate_questions）：从答案反推用户可能问的问题

所有方法均异步执行，内部统一处理重试、异常和输出解析。
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

from loguru import logger


# ---------------------------------------------------------------------------
# LLM 调用协议
# ---------------------------------------------------------------------------


@runtime_checkable
class JudgeLLMProtocol(Protocol):
    """
    评判器使用的 LLM 接口协议。

    任何实现了 ``acomplete(messages, temperature) -> str`` 的对象都可作为评判 LLM 传入，
    不依赖具体模型类型（LangChain / OpenAI SDK / 项目内部适配器均可）。

    示例::

        class MyLLM:
            async def acomplete(self, messages: list[dict], temperature: float = 0.0) -> str:
                ...

        judge = LLMJudge(llm=MyLLM())
    """

    async def acomplete(
        self, messages: list[dict[str, str]], temperature: float = 0.0
    ) -> str:
        """
        异步完成接口。

        :param messages: OpenAI 风格的消息列表 ``[{"role": "...", "content": "..."}]``
        :param temperature: 采样温度，评判场景推荐 0.0（确定性输出）
        :returns: LLM 返回的文本
        """
        ...


# ---------------------------------------------------------------------------
# LLM 评判器
# ---------------------------------------------------------------------------


class LLMJudge:
    """
    RAGAS 评估专用 LLM 评判器。

    封装了数值打分、分类判定、陈述拆分、反向问题生成等基础能力，
    为上层指标计算（Faithfulness / Context Precision / Answer Relevancy 等）提供原子操作。

    使用示例::

        judge = LLMJudge(llm=my_llm_adapter, max_retries=3)
        score = await judge.score("判断以下回答是否忠实于上下文...", temperature=0.0)
        claims = await judge.extract_claims("人工智能是计算机科学的一个分支...")
    """

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(
        self,
        llm: JudgeLLMProtocol | None = None,
        max_retries: int = 3,
        default_temperature: float = 0.0,
    ) -> None:
        """
        :param llm: 实现 ``acomplete`` 的 LLM 对象，若为 None 则所有方法返回默认值
        :param max_retries: LLM 调用失败时的最大重试次数
        :param default_temperature: 默认温度，评判场景推荐 0.0 以保证一致性
        """
        self._llm = llm
        self._max_retries = max_retries
        self._default_temperature = default_temperature

    # ------------------------------------------------------------------
    # 数值打分
    # ------------------------------------------------------------------

    async def score(
        self,
        prompt: str,
        temperature: float | None = None,
        *,
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> float:
        """
        让 LLM 对某个维度打出数值分数。

        LLM 被要求输出一个纯数字（如 ``0.85``），内部会做范围裁剪。

        :param prompt: 评分 prompt，需在末尾明确要求输出数字
        :param temperature: 温度，None 则使用默认值
        :param min_val: 分数下限
        :param max_val: 分数上限
        :returns: 裁剪后的分数
        """
        if self._llm is None:
            logger.warning("LLMJudge: 未配置 LLM，score 返回默认值 {}", (min_val + max_val) / 2)
            return (min_val + max_val) / 2

        temp = temperature if temperature is not None else self._default_temperature

        for attempt in range(1, self._max_retries + 1):
            try:
                raw = await self._llm.acomplete(
                    [{"role": "user", "content": prompt}],
                    temperature=temp,
                )
                score_val = self._parse_number(raw, default=(min_val + max_val) / 2)
                clamped = max(min_val, min(max_val, score_val))
                if attempt > 1:
                    logger.debug("LLMJudge.score 第 {} 次重试成功", attempt)
                return clamped
            except Exception as exc:
                logger.warning("LLMJudge.score 第 {}/{} 次失败: {}", attempt, self._max_retries, exc)
                if attempt >= self._max_retries:
                    return (min_val + max_val) / 2

        return (min_val + max_val) / 2

    # ------------------------------------------------------------------
    # 分类判定
    # ------------------------------------------------------------------

    async def classify(
        self,
        prompt: str,
        labels: list[str],
        temperature: float | None = None,
    ) -> str:
        """
        让 LLM 从候选标签列表中选择最匹配的一个。

        :param prompt: 分类 prompt，需在末尾要求从给定标签中选择
        :param labels: 候选标签列表，如 ``["相关", "不相关"]``
        :param temperature: 温度
        :returns: LLM 输出的标签（若无法匹配则返回第一个标签）
        """
        if self._llm is None:
            logger.warning("LLMJudge: 未配置 LLM，classify 返回默认标签 {}", labels[0])
            return labels[0]

        temp = temperature if temperature is not None else self._default_temperature

        for attempt in range(1, self._max_retries + 1):
            try:
                raw = await self._llm.acomplete(
                    [{"role": "user", "content": prompt}],
                    temperature=temp,
                )
                best = self._best_label(raw.strip(), labels)
                if attempt > 1:
                    logger.debug("LLMJudge.classify 第 {} 次重试成功 → {}", attempt, best)
                return best
            except Exception as exc:
                logger.warning("LLMJudge.classify 第 {}/{} 次失败: {}", attempt, self._max_retries, exc)
                if attempt >= self._max_retries:
                    return labels[0]

        return labels[0]

    # ------------------------------------------------------------------
    # 原子陈述拆分
    # ------------------------------------------------------------------

    async def extract_claims(
        self,
        text: str,
        temperature: float | None = None,
    ) -> list[str]:
        """
        将一段文本拆分为原子陈述（claims）。

        每个 claim 是一个独立的、可被单独验证的简单陈述句。
        用于 Faithfulness 指标计算：判断每个 claim 是否被检索上下文支撑。

        :param text: 待拆分的完整文本（通常为 RAG 回答）
        :param temperature: 温度
        :returns: 原子陈述列表，若 LLM 不可用/失败则返回整句作为单个 claim
        """
        if self._llm is None or not text.strip():
            return [text] if text.strip() else []

        prompt = (
            "你是一个文本分析助手。请将以下文本拆分为独立的「原子陈述」列表。\n"
            "每个陈述必须是一个简单、独立、可被单独验证的事实句。\n"
            "输出格式：一个 JSON 字符串数组。\n\n"
            f"文本：{text.strip()[:2000]}\n\n"
            "请只输出 JSON 数组，不要有其他内容："
        )

        temp = temperature if temperature is not None else self._default_temperature

        for attempt in range(1, self._max_retries + 1):
            try:
                raw = await self._llm.acomplete(
                    [{"role": "user", "content": prompt}],
                    temperature=temp,
                )
                claims = self._parse_json_list(raw)
                if claims:
                    return claims
            except Exception as exc:
                logger.warning("LLMJudge.extract_claims 第 {}/{} 次失败: {}", attempt, self._max_retries, exc)

        # Fallback：按中文句号/分号简单拆分
        fallback = [s.strip() + "。" for s in re.split(r"[。；;]", text) if s.strip()]
        return fallback if fallback else [text.strip()]

    # ------------------------------------------------------------------
    # 反向问题生成
    # ------------------------------------------------------------------

    async def generate_questions(
        self,
        answer: str,
        n: int = 3,
        temperature: float | None = 0.3,
    ) -> list[str]:
        """
        从答案反向生成用户可能提出的问题。

        用于 AnswerRelevancy 指标：将生成的 questions 与原始 query 计算语义相似度，
        判断答案是否切题。

        :param answer: RAG 生成的答案
        :param n: 生成的问题数量（推荐 3-5）
        :param temperature: 反向生成时可使用稍高温度增加多样性
        :returns: 生成的问题列表，LLM 不可用时返回空列表
        """
        if self._llm is None:
            logger.warning("LLMJudge: 未配置 LLM，generate_questions 返回空列表")
            return []

        prompt = (
            "你是一个测试问题生成器。请根据以下「答案」，"
            f"反向生成 {n} 个用户可能提出的「问题」。\n"
            "这些问题应该覆盖答案中的关键信息点，"
            "且每个问题都应以该答案作为合理回复。\n\n"
            f"答案：{answer.strip()[:2000]}\n\n"
            f"请输出一个 JSON 字符串数组，包含恰好 {n} 个问题，不要有其他内容："
        )

        temp = temperature if temperature is not None else 0.3

        for attempt in range(1, self._max_retries + 1):
            try:
                raw = await self._llm.acomplete(
                    [{"role": "user", "content": prompt}],
                    temperature=temp,
                )
                questions = self._parse_json_list(raw)
                if questions and len(questions) >= 1:
                    return questions[:n]
            except Exception as exc:
                logger.warning("LLMJudge.generate_questions 第 {}/{} 次失败: {}", attempt, self._max_retries, exc)

        return []

    # ------------------------------------------------------------------
    # 私有工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_number(raw: str, default: float = 0.5) -> float:
        """
        从 LLM 原始输出中提取数值。

        支持格式：``0.85``、``分数：0.9``、``Score: 0.75/1`` 等。
        """
        # 尝试匹配浮点数
        matches = re.findall(r"([0-9]*\.?[0-9]+)", raw)
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                pass
        return default

    @staticmethod
    def _best_label(raw: str, labels: list[str]) -> str:
        """
        从 LLM 输出中匹配最可能的标签。

        优先精确匹配，其次子串匹配，最后返回默认标签。
        """
        raw_lower = raw.lower().strip()
        # 精确匹配
        for label in labels:
            if label.lower() == raw_lower:
                return label
        # 子串匹配（选择最长匹配）
        best, best_len = labels[0], 0
        for label in labels:
            if label.lower() in raw_lower and len(label) > best_len:
                best, best_len = label, len(label)
        return best

    @staticmethod
    def _parse_json_list(raw: str) -> list[str]:
        """
        从 LLM 输出中解析 JSON 字符串数组。

        容忍前后额外文本，自动提取第一个 JSON 数组。
        """
        # 尝试直接解析
        try:
            data = json.loads(raw.strip())
            if isinstance(data, list):
                return [str(item) for item in data]
        except (json.JSONDecodeError, TypeError):
            pass

        # 用正则提取 JSON 数组片段
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return [str(item) for item in data]
            except (json.JSONDecodeError, TypeError):
                pass

        # 按行解析（LLM 有时会逐行输出，每行一个带数字编号的陈述）
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        # 去编号：移除开头的 "1." "2." "-" 等
        cleaned: list[str] = []
        for line in lines:
            cleaned_line = re.sub(r"^[\d]+[\.\)、]\s*", "", line).strip()
            cleaned_line = re.sub(r"^[-*]\s*", "", cleaned_line).strip()
            if cleaned_line and len(cleaned_line) > 3:
                cleaned.append(cleaned_line)
        return cleaned if len(cleaned) >= 2 else []
