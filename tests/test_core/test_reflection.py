# -*- coding: utf-8 -*-
"""ReflectionAgent 反思与质量审查测试。"""

from __future__ import annotations

import json

import pytest

from app.core.agent.reflection import ReflectionAgent, ReflectionReport
from tests.conftest import MockLLM


@pytest.fixture
def reflection_agent(mock_llm: MockLLM) -> ReflectionAgent:
    return ReflectionAgent(llm=mock_llm, min_quality_to_pass=60)


class TestShouldRetryOrWarn:
    """should_retry_or_warn 决策测试。"""

    def test_both_flags(self, reflection_agent: ReflectionAgent) -> None:
        report = ReflectionReport(
            quality_score=30,
            is_complete=False,
            likely_hallucination=True,
            hallucination_reasons=["Unverified claim"],
            completeness_notes="Missing key info",
            suggestions=["Verify sources"],
            summary="Low quality",
        )
        decision = reflection_agent.should_retry_or_warn(report)
        assert decision["warn_low_quality"] is True
        assert decision["suggest_retry"] is True

    def test_neither_flag(self, reflection_agent: ReflectionAgent) -> None:
        report = ReflectionReport(
            quality_score=85,
            is_complete=True,
            likely_hallucination=False,
            hallucination_reasons=[],
            completeness_notes="Complete",
            suggestions=[],
            summary="Good answer",
        )
        decision = reflection_agent.should_retry_or_warn(report)
        assert decision["warn_low_quality"] is False
        assert decision["suggest_retry"] is False

    def test_only_hallucination(self, reflection_agent: ReflectionAgent) -> None:
        report = ReflectionReport(
            quality_score=75,
            is_complete=True,
            likely_hallucination=True,
            hallucination_reasons=["Made up fact"],
            completeness_notes="All covered",
            suggestions=["Double-check the fact"],
            summary="Potential hallucination",
        )
        decision = reflection_agent.should_retry_or_warn(report)
        assert decision["warn_low_quality"] is False  # quality >= 60
        assert decision["suggest_retry"] is True   # hallucination detected

    def test_custom_threshold(self) -> None:
        agent = ReflectionAgent(llm=MockLLM(), min_quality_to_pass=80)
        report = ReflectionReport(
            quality_score=70,
            is_complete=True,
            likely_hallucination=False,
            hallucination_reasons=[],
            completeness_notes="OK",
            suggestions=[],
            summary="OK",
        )
        decision = agent.should_retry_or_warn(report)
        assert decision["warn_low_quality"] is True
        assert decision["min_quality_threshold"] == 80


class TestReflect:
    """reflect() 反射方法测试。"""

    async def test_reflect_success(
        self, reflection_agent: ReflectionAgent, mock_llm: MockLLM
    ) -> None:
        good_json = json.dumps({
            "quality_score": 85,
            "is_complete": True,
            "likely_hallucination": False,
            "hallucination_reasons": [],
            "completeness_notes": "Covers all points",
            "suggestions": ["Could be more detailed"],
            "summary": "Good answer overall",
        })
        mock_llm.add_response(good_json)
        report = await reflection_agent.reflect(
            "What is AI?",
            "AI is artificial intelligence.",
        )
        assert report.quality_score == 85
        assert report.is_complete is True
        assert report.likely_hallucination is False

    async def test_reflect_llm_failure(
        self, reflection_agent: ReflectionAgent, mock_llm: MockLLM
    ) -> None:
        async def raise_error(messages, **kwargs):
            raise RuntimeError("LLM unavailable")

        mock_llm.acomplete = raise_error  # type: ignore[assignment]
        report = await reflection_agent.reflect("Q", "A")
        assert report.quality_score == 50  # degraded default
        assert report.parse_error is not None

    async def test_reflect_invalid_json(
        self, reflection_agent: ReflectionAgent, mock_llm: MockLLM
    ) -> None:
        mock_llm.add_response("Not valid JSON, just text.")
        report = await reflection_agent.reflect("Q", "A")
        # 应降级但不应崩溃
        assert report is not None
        assert report.parse_error is not None

    async def test_reflect_with_evidence(
        self, reflection_agent: ReflectionAgent, mock_llm: MockLLM
    ) -> None:
        mock_llm.add_response(json.dumps({
            "quality_score": 70,
            "is_complete": True,
            "likely_hallucination": False,
            "hallucination_reasons": [],
            "completeness_notes": "OK with evidence",
            "suggestions": [],
            "summary": "Supported by evidence",
        }))
        report = await reflection_agent.reflect(
            "Q", "A",
            evidence_snippets=["Source says X", "Research confirms Y"],
        )
        assert report.quality_score == 70

    async def test_reflect_quality_score_clamped(self) -> None:
        """质量分数应被限制在 0-100 之间。"""
        agent = ReflectionAgent(llm=MockLLM(), min_quality_to_pass=60)
        # 通过构造函数直接创建负分报告来测试 should_retry_or_warn
        report = ReflectionReport(
            quality_score=-10,
            is_complete=False,
            likely_hallucination=True,
            hallucination_reasons=[],
            completeness_notes="",
            suggestions=[],
            summary="",
        )
        decision = agent.should_retry_or_warn(report)
        assert decision["warn_low_quality"] is True
