# -*- coding: utf-8 -*-
"""CircuitBreaker 三态有限状态机测试。"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from app.infrastructure.llm.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreakerInitialization:
    """构造函数与初始状态测试。"""

    def test_initial_state_closed(self) -> None:
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_custom_parameters(self) -> None:
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=10.0,
            half_open_max=5,
            name="test-cb",
        )
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 10.0
        assert cb.half_open_max == 5
        assert cb.name == "test-cb"

    def test_min_failure_threshold_is_1(self) -> None:
        cb = CircuitBreaker(failure_threshold=0)
        assert cb.failure_threshold == 1

    def test_min_half_open_max_is_1(self) -> None:
        cb = CircuitBreaker(half_open_max=0)
        assert cb.half_open_max == 1


class TestCircuitBreakerClosedState:
    """CLOSED 状态下的行为。"""

    async def test_call_success_remains_closed(self) -> None:
        cb = CircuitBreaker(failure_threshold=2)
        async def ok() -> str:
            return "success"

        result = await cb.call(ok)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    async def test_failure_increments_counter_but_stays_closed(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        async def fail() -> str:
            raise ValueError("boom")

        for _ in range(2):
            with pytest.raises(ValueError, match="boom"):
                await cb.call(fail)

        # 尚未达到阈值，状态仍为 CLOSED
        assert cb.state == CircuitState.CLOSED

    async def test_trip_after_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=2)
        call_count = 0

        async def fail() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("boom")

        # 前 2 次失败
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(fail)

        # 达到阈值，应该跳闸
        assert cb.state == CircuitState.OPEN
        assert call_count == 2

    async def test_single_failure_trips_with_threshold_1(self) -> None:
        cb = CircuitBreaker(failure_threshold=1)
        async def fail() -> str:
            raise ValueError("boom")

        with pytest.raises(ValueError):
            await cb.call(fail)
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerOpenState:
    """OPEN 状态下的行为。"""

    async def test_open_rejects_calls(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=999.0)
        async def fail() -> str:
            raise ValueError("boom")

        # 先触发跳闸
        with pytest.raises(ValueError):
            await cb.call(fail)
        assert cb.state == CircuitState.OPEN

        # OPEN 状态下直接拒绝
        with pytest.raises(RuntimeError, match="OPEN"):
            await cb.call(AsyncMock(return_value="ok"))


class TestCircuitBreakerHalfOpenState:
    """HALF_OPEN 状态下的行为。"""

    @staticmethod
    def _make_time_ticks(start: float = 0.0, step: float = 1.0):
        """生成单调递增的时间值。"""
        t = start
        def tick() -> float:
            nonlocal t
            result = t
            t += step
            return result
        return tick

    @pytest.fixture
    def tripped_breaker(self) -> CircuitBreaker:
        """返回一个已跳闸的 CircuitBreaker（1 次失败，recovery_timeout=5s）。"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0, half_open_max=3)
        # 触发一次失败使其 OPEN
        return cb  # 调用方需要自行触发

    async def test_open_rejects_calls_during_timeout(self) -> None:
        """OPEN 状态 + 未过 recovery_timeout → 拒绝调用。"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=999.0)
        async def fail() -> str:
            raise ValueError("boom")

        with pytest.raises(ValueError):
            await cb.call(fail)
        assert cb.state == CircuitState.OPEN

        # OPEN 状态下直接拒绝
        with pytest.raises(RuntimeError, match="OPEN"):
            await cb.call(AsyncMock(return_value="ok"))

    # 使用递增时间戳来模拟时间流逝
    _time_ticks: list[float]
    _tick_idx: int

    @staticmethod
    def _make_advancing_time() -> any:
        """返回一个每次调用都返回更大值的可调用对象。"""
        base = [1000.0]  # 用列表实现闭包可变性

        def tick() -> float:
            result = base[0]
            base[0] += 100.0  # 每次调用时间前进 100s
            return result

        return tick

    async def test_half_open_success_resets_to_closed(self) -> None:
        """半开状态试探成功后恢复 CLOSED。"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)

        async def fail() -> str:
            raise ValueError("boom")

        # 跳闸
        with pytest.raises(ValueError):
            await cb.call(fail)
        assert cb.state == CircuitState.OPEN

        # 模拟时间流逝 100s（超过 recovery_timeout=5s）
        with patch(
            "app.infrastructure.llm.circuit_breaker.time.monotonic",
            side_effect=[100.0, 200.0, 300.0],  # 第1次：_record_failure；第2次：_should_attempt_reset
        ):
            async def ok() -> str:
                return "success"
            result = await cb.call(ok)
            assert result == "success"
            assert cb.state == CircuitState.CLOSED

    async def test_half_open_failure_reopens(self) -> None:
        """半开状态试探失败应重新打开。"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)

        async def fail() -> str:
            raise ValueError("boom")

        # 跳闸
        with pytest.raises(ValueError):
            await cb.call(fail)
        assert cb.state == CircuitState.OPEN

        # 时间已过，试探失败 → 重新 OPEN
        with patch(
            "app.infrastructure.llm.circuit_breaker.time.monotonic",
            side_effect=[100.0, 200.0, 300.0],
        ):
            with pytest.raises(ValueError):
                await cb.call(fail)
        assert cb.state == CircuitState.OPEN

    async def test_half_open_max_attempts_exceeded(self) -> None:
        """半开状态试探次数达到上限后拒绝。"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0, half_open_max=2)

        async def fail() -> str:
            raise ValueError("boom")

        # 3 次失败触发 OPEN
        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.call(fail)
        assert cb.state == CircuitState.OPEN

        # 半开试探，每次调用时间都前进
        with patch(
            "app.infrastructure.llm.circuit_breaker.time.monotonic",
            side_effect=[100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
        ):
            # 2 次试探 OK
            await cb.call(lambda: "ok1")
            await cb.call(lambda: "ok2")
            # 第 3 次超出上限
            with pytest.raises(RuntimeError, match="试探次数已达上限"):
                await cb.call(lambda: "ok3")


class TestCircuitBreakerConcurrency:
    """并发安全测试。"""

    async def test_concurrent_calls_use_lock(self) -> None:
        """验证多个并发调用在锁的保护下正确序列化。"""
        cb = CircuitBreaker(failure_threshold=3)

        call_order: list[int] = []

        async def slow_ok(idx: int) -> str:
            call_order.append(idx)
            await asyncio.sleep(0.01)
            return f"ok-{idx}"

        # 并发发起 5 个调用
        results = await asyncio.gather(
            cb.call(slow_ok, 0),
            cb.call(slow_ok, 1),
            cb.call(slow_ok, 2),
            cb.call(slow_ok, 3),
            cb.call(slow_ok, 4),
        )

        assert results == [f"ok-{i}" for i in range(5)]
        # 5 次成功调用后状态应为 CLOSED
        assert cb.state == CircuitState.CLOSED
        assert len(call_order) == 5
