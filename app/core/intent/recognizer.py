# -*- coding: utf-8 -*-
"""意图识别：树形分类、置信度与澄清引导。"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    """意图识别结果。"""

    intent: str = Field(description="主意图标签")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度 0~1")
    slots: dict[str, Any] = Field(default_factory=dict, description="槽位信息")
    sub_intent: str | None = Field(default=None, description="子意图（树形第二层）")
    rationale: str = Field(default="", description="简要理由（可给模型或日志用）")


# 简易树形意图：根 -> 子意图（关键词规则，可替换为分类模型）
_INTENT_TREE: dict[str, dict[str, list[str]]] = {
    "问答": {
        "知识问答": [
            "什么", "是什么", "为什么", "如何", "怎么", "哪些", "多少", "谁",
            "解释", "含义", "属性", "技能", "精灵", "图鉴",
        ],
        "闲聊": ["你好", "谢谢", "再见", "聊天"],
    },
    "任务": {
        "搜索": ["搜索", "查一下", "帮我找", "search"],
        "计算": ["计算", "等于", "算", "加", "减"],
        "数据库": ["sql", "查询", "表", "统计"],
    },
    "文档": {
        "上传": ["上传", "文档", "pdf"],
        "总结": ["总结", "摘要", "概括"],
    },
}


class IntentRecognizer:
    """意图识别器：树形分类 + 置信度 + 引导澄清。"""

    def __init__(self, confidence_threshold: float = 0.55) -> None:
        self._threshold = confidence_threshold

    @property
    def confidence_threshold(self) -> float:
        """置信度阈值：低于该值时建议澄清。"""
        return self._threshold

    def _score_branch(self, query: str) -> tuple[str, str | None, float, str]:
        """返回 (根意图, 子意图, 置信度, 理由)。"""
        q = query.strip().lower()
        if re.search(r"\d+\s*[-+*/%^]\s*\d+", q) or any(
            word in q for word in ("计算", "等于", "加", "减", "乘", "除", "平方", "开方")
        ):
            return "任务", "计算", 0.9, "命中数学表达式或计算关键词"
        if any(word in q for word in ("搜索", "查一下", "帮我找", "search")):
            return "任务", "搜索", 0.85, "命中搜索关键词"
        if any(word in q for word in ("sql", "数据库", "统计", "查询表")):
            return "任务", "数据库", 0.85, "命中数据库查询关键词"
        if any(word in q for word in ("知识库", "文档", "资料里", "已上传")):
            return "问答", "知识问答", 0.85, "命中知识库关键词"
        if any(word in q for word in ("是什么", "有哪些", "哪些", "多少", "技能", "属性", "精灵", "图鉴")):
            return "问答", "知识问答", 0.75, "命中知识问答关键词"
        best_root = "未知"
        best_child: str | None = None
        best_hits = 0
        total_kw = 0

        for root, children in _INTENT_TREE.items():
            for child, kws in children.items():
                hits = sum(1 for kw in kws if kw.lower() in q)
                if hits > best_hits:
                    best_hits = hits
                    best_root = root
                    best_child = child
                total_kw += len(kws)

        if best_hits == 0:
            # 无关键词命中：根据长度与问号弱推断
            conf = 0.35 if "?" in q or "？" in q else 0.25
            return "问答", "知识问答", conf, "未命中关键词，弱规则推断"

        conf = min(0.95, 0.45 + 0.12 * best_hits)
        rationale = f"关键词命中 {best_hits} 次，归类为 {best_root}/{best_child}"
        return best_root, best_child, conf, rationale

    async def recognize(self, query: str, context: dict | None = None) -> IntentResult:
        """识别用户意图；context 可包含历史轮次等（当前规则引擎未使用）。"""
        _ = context
        root, child, conf, why = self._score_branch(query)
        slots: dict[str, Any] = {}
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", query)
        if nums:
            slots["numbers"] = nums[:5]

        result = IntentResult(
            intent=root,
            confidence=conf,
            slots=slots,
            sub_intent=child,
            rationale=why,
        )
        logger.info(
            "意图识别 query_preview={} -> {} / {} conf={}",
            query[:100],
            root,
            child,
            conf,
        )
        return result

    async def recognize_with_llm(
        self,
        query: str,
        context: dict | None = None,
        llm: Any | None = None,
    ) -> IntentResult:
        """级联识别：高置信规则优先，低置信时使用 LLM 结构化分类。"""
        rule_result = await self.recognize(query, context=context)
        if rule_result.confidence >= 0.8 or llm is None or len(query.strip()) <= 3:
            return rule_result

        prompt = self._build_llm_prompt(query, context=context, rule_result=rule_result)
        try:
            raw = await llm.acomplete([
                {"role": "system", "content": "你是 Agent 意图识别器，只输出合法 JSON。"},
                {"role": "user", "content": prompt},
            ], temperature=0.0)
            llm_result = self._parse_llm_result(raw)
            if llm_result.confidence >= rule_result.confidence:
                return llm_result
        except Exception as exc:
            logger.warning("LLM 意图识别失败，回退规则结果: {}", exc)
        return rule_result

    async def clarify(self, query: str, intent_result: IntentResult) -> str:
        """置信度不足时生成澄清提示语。"""
        if intent_result.confidence >= self._threshold:
            return ""

        return (
            "我不太确定您的具体需求。"
            f"您是想了解「{intent_result.intent}」相关（当前置信度 {intent_result.confidence:.2f}）吗？"
            "请补充场景、对象或期望的输出格式（例如：只要结论 / 需要步骤）。"
        )

    def _build_llm_prompt(
        self,
        query: str,
        context: dict | None,
        rule_result: IntentResult,
    ) -> str:
        messages = (context or {}).get("messages") or []
        recent = messages[-4:] if isinstance(messages, list) else []
        return json.dumps({
            "task": "请根据用户问题判断 Agent 路由意图。",
            "labels": {
                "intent": ["问答", "任务", "文档", "未知"],
                "sub_intent": ["知识问答", "闲聊", "搜索", "计算", "数据库", "上传", "总结", "未知"],
            },
            "routing_hint": {
                "知识问答": "知识库、知识图谱、精灵图鉴、事实解释、属性技能查询",
                "搜索": "需要联网查最新信息",
                "计算": "数学或表达式计算",
                "数据库": "SQL、表、统计、数据库查询",
                "上传": "上传或管理文件",
                "总结": "摘要、概括、总结文档",
                "闲聊": "问候、感谢、普通聊天",
            },
            "output_schema": {
                "intent": "问答|任务|文档|未知",
                "sub_intent": "知识问答|闲聊|搜索|计算|数据库|上传|总结|未知",
                "confidence": "0到1之间的数字",
                "slots": "对象、实体、数字、约束等键值",
                "rationale": "一句话理由",
            },
            "rule_baseline": rule_result.model_dump(),
            "recent_messages": recent,
            "query": query,
        }, ensure_ascii=False)

    def _parse_llm_result(self, raw: str) -> IntentResult:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        try:
            payload = json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                raise ValueError("LLM 未返回 JSON")
            payload = json.loads(match.group(0))

        intent = str(payload.get("intent") or "未知")
        sub_intent = str(payload.get("sub_intent") or "未知")
        try:
            confidence = float(payload.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(confidence, 1.0))
        slots = payload.get("slots") if isinstance(payload.get("slots"), dict) else {}
        rationale = str(payload.get("rationale") or "LLM 结构化分类")
        return IntentResult(
            intent=intent,
            confidence=confidence,
            slots=slots,
            sub_intent=sub_intent,
            rationale=rationale,
        )
