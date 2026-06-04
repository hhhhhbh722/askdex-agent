# -*- coding: utf-8 -*-
"""Tracer 全链路追踪测试。"""

from __future__ import annotations

import threading

import pytest

from app.infrastructure.trace.tracer import Tracer


@pytest.fixture
def tracer() -> Tracer:
    return Tracer(max_traces=10)


class TestTraceLifecycle:
    """追踪 span 生命周期测试。"""

    def test_start_trace_creates_record(self, tracer: Tracer) -> None:
        trace_id = "trace-001"
        span = tracer.start_trace(trace_id, "test_op")

        assert span.trace_id == trace_id
        assert span.operation == "test_op"
        assert span.start_time is not None

    def test_start_trace_appends_span_to_existing(self, tracer: Tracer) -> None:
        trace_id = "trace-002"
        span1 = tracer.start_trace(trace_id, "op1")
        span2 = tracer.start_trace(trace_id, "op2")

        trace = tracer.get_trace(trace_id)
        assert trace is not None
        assert len(trace.spans) == 2

    def test_child_span_has_parent(self, tracer: Tracer) -> None:
        trace_id = "trace-003"
        parent = tracer.start_trace(trace_id, "parent_op")
        child = tracer.start_child_span(trace_id, "child_op")

        assert child.parent_span_id == parent.span_id

    def test_child_span_implicit_create_trace(self, tracer: Tracer) -> None:
        """在无现有 trace 时，start_child_span 应隐式创建 trace。"""
        child = tracer.start_child_span("trace-new", "child_op")
        assert child is not None
        assert child.trace_id == "trace-new"

    def test_end_span_sets_result(self, tracer: Tracer) -> None:
        trace_id = "trace-004"
        span = tracer.start_trace(trace_id, "op")
        result = {"status": "ok", "count": 42}

        tracer.end_span(span, result=result)
        assert span.result == result
        assert span.end_time is not None

    def test_end_span_sets_error(self, tracer: Tracer) -> None:
        trace_id = "trace-005"
        span = tracer.start_trace(trace_id, "op")
        tracer.end_span(span, error="Something went wrong")
        assert span.error == "Something went wrong"

    def test_end_span_unknown_trace_warns(self, tracer: Tracer) -> None:
        """对不存在 trace 的 span 调用 end_span 不应崩溃。"""
        # 创建 span 后手动清空内部状态来模拟
        trace_id = "trace-006"
        span = tracer.start_trace(trace_id, "op")
        # 直接删除 trace 记录
        del tracer._traces[trace_id]

        # 不应抛出异常
        tracer.end_span(span, result={"status": "ok"})


class TestTraceRetrieval:
    """追踪读取测试。"""

    def test_get_trace_returns_copy(self, tracer: Tracer) -> None:
        trace_id = "trace-007"
        tracer.start_trace(trace_id, "op1")
        tracer.start_trace(trace_id, "op2")

        trace = tracer.get_trace(trace_id)
        assert trace is not None
        assert len(trace.spans) == 2

    def test_get_trace_unknown_returns_none(self, tracer: Tracer) -> None:
        assert tracer.get_trace("nonexistent") is None

    def test_get_trace_copy_is_immutable(self, tracer: Tracer) -> None:
        trace_id = "trace-008"
        tracer.start_trace(trace_id, "op1")

        trace = tracer.get_trace(trace_id)
        assert trace is not None
        trace.spans.append(trace.spans[0])  # 修改副本

        # 原始应不受影响
        original = tracer.get_trace(trace_id)
        assert original is not None
        assert len(original.spans) == 1


class TestTraceTrimming:
    """追踪数量限制与裁剪测试。"""

    def test_trim_when_exceeding_max_traces(self, tracer: Tracer) -> None:
        # 创建超过 max_traces 的 trace
        for i in range(15):
            tracer.start_trace(f"trace-{i:03d}", "op")

        assert len(tracer._traces) <= 10

    def test_max_traces_configurable(self) -> None:
        t = Tracer(max_traces=3)
        for i in range(10):
            t.start_trace(f"trace-{i}", "op")
        assert len(t._traces) <= 3


class TestTraceConcurrency:
    """并发安全测试。"""

    def test_concurrent_trace_creation(self, tracer: Tracer) -> None:
        """多线程并发创建 trace 不应损坏数据结构。"""
        errors: list[Exception] = []

        def create_traces(prefix: str) -> None:
            try:
                for i in range(20):
                    tracer.start_trace(f"{prefix}-{i}", "op")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=create_traces, args=(f"t{n}",))
            for n in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
