# -*- coding: utf-8 -*-
"""IntentRecognizer 意图识别测试。"""

from __future__ import annotations

import pytest

from app.core.intent.recognizer import IntentRecognizer


@pytest.fixture
def recognizer() -> IntentRecognizer:
    return IntentRecognizer(confidence_threshold=0.55)


class TestIntentRecognition:
    """基本意图分类测试。"""

    async def test_knowledge_qa_triggered_by_question(self, recognizer: IntentRecognizer) -> None:
        result = await recognizer.recognize("什么是人工智能？")
        assert result.intent == "问答"
        assert result.sub_intent == "知识问答"

    async def test_chitchat_triggered_by_greeting(self, recognizer: IntentRecognizer) -> None:
        result = await recognizer.recognize("你好啊")
        assert result.intent == "问答"
        assert result.sub_intent == "闲聊"

    async def test_search_triggered_by_keyword(self, recognizer: IntentRecognizer) -> None:
        result = await recognizer.recognize("帮我搜索最新新闻")
        assert result.intent == "任务"
        assert result.sub_intent == "搜索"

    async def test_calculation_triggered_by_math(self, recognizer: IntentRecognizer) -> None:
        result = await recognizer.recognize("计算 123 + 456")
        assert result.intent == "任务"
        assert result.sub_intent == "计算"

    async def test_database_triggered_by_sql(self, recognizer: IntentRecognizer) -> None:
        result = await recognizer.recognize("查询用户表中所有记录")
        assert result.intent == "任务"
        assert result.sub_intent == "数据库"

    async def test_document_upload_triggered(self, recognizer: IntentRecognizer) -> None:
        result = await recognizer.recognize("上传一个PDF文档")
        assert result.intent == "文档"
        assert result.sub_intent == "上传"

    async def test_document_summary_triggered(self, recognizer: IntentRecognizer) -> None:
        result = await recognizer.recognize("总结一下这篇文章的主要内容")
        assert result.intent == "文档"
        assert result.sub_intent == "总结"


class TestIntentConfidence:
    """置信度相关测试。"""

    async def test_keyword_match_bumps_confidence(self, recognizer: IntentRecognizer) -> None:
        result = await recognizer.recognize("搜索人工智能最新进展")
        # 关键词命中应使置信度 > 0.45
        assert result.confidence > 0.45

    async def test_unknown_query_low_confidence(self, recognizer: IntentRecognizer) -> None:
        result = await recognizer.recognize("xyzzy")
        assert result.confidence < 0.5

    async def test_question_mark_boosts_confidence(self, recognizer: IntentRecognizer) -> None:
        result = await recognizer.recognize("xyz?")
        # "?" 触发弱推断，置信度提高
        assert result.confidence >= 0.30

    async def test_empty_query(self, recognizer: IntentRecognizer) -> None:
        result = await recognizer.recognize("")
        # 空字符串也能返回结果
        assert result.intent in ("问答", "未知")


class TestNumberExtraction:
    """槽位数值提取测试。"""

    async def test_extract_numbers_from_query(self, recognizer: IntentRecognizer) -> None:
        result = await recognizer.recognize("帮我算一下 23 加上 45.6 再加 7")
        assert "numbers" in result.slots
        assert "23" in result.slots["numbers"]
        assert "45.6" in result.slots["numbers"]
        assert "7" in result.slots["numbers"]


class TestClarification:
    """低置信度引导澄清测试。"""

    async def test_high_confidence_no_clarification(self, recognizer: IntentRecognizer) -> None:
        intent_result = await recognizer.recognize("搜索人工智能")
        clarification = await recognizer.clarify("搜索人工智能", intent_result)
        # 高置信度时不生成澄清
        assert clarification == ""

    async def test_low_confidence_generates_clarification(self, recognizer: IntentRecognizer) -> None:
        recognizer_low = IntentRecognizer(confidence_threshold=0.95)
        intent_result = await recognizer_low.recognize("xyz something")
        # 无关键词命中，置信度应很低（阈值 = 0.95）
        clarification = await recognizer_low.clarify("xyz something", intent_result)
        assert len(clarification) > 0

    async def test_confidence_threshold_property(self, recognizer: IntentRecognizer) -> None:
        r = IntentRecognizer(confidence_threshold=0.7)
        assert r.confidence_threshold == 0.7
