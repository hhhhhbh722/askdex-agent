# -*- coding: utf-8 -*-
"""CalculatorTool AST 安全求值测试。"""

from __future__ import annotations

import pytest

from app.core.tools.builtin.calculator import CalculatorTool


@pytest.fixture
def calc() -> CalculatorTool:
    return CalculatorTool()


class TestCalculatorBasicOperations:
    """基本算术运算。"""

    async def test_addition(self, calc: CalculatorTool) -> None:
        result = await calc.execute(expression="2 + 3")
        assert str(result) == "5.0"

    async def test_subtraction(self, calc: CalculatorTool) -> None:
        result = await calc.execute(expression="10 - 4")
        assert str(result) == "6.0"

    async def test_multiplication(self, calc: CalculatorTool) -> None:
        result = await calc.execute(expression="5 * 6")
        assert str(result) == "30.0"

    async def test_division(self, calc: CalculatorTool) -> None:
        result = await calc.execute(expression="15 / 3")
        assert str(result) == "5.0"

    async def test_power(self, calc: CalculatorTool) -> None:
        result = await calc.execute(expression="2 ** 10")
        assert str(result) == "1024.0"

    async def test_modulo(self, calc: CalculatorTool) -> None:
        result = await calc.execute(expression="10 % 3")
        assert str(result) == "1.0"

    async def test_negative_number(self, calc: CalculatorTool) -> None:
        result = await calc.execute(expression="-5")
        assert str(result) == "-5.0"

    async def test_unary_plus(self, calc: CalculatorTool) -> None:
        result = await calc.execute(expression="+3")
        assert str(result) == "3.0"


class TestCalculatorPrecedence:
    """运算符优先级测试。"""

    async def test_multiplication_before_addition(self, calc: CalculatorTool) -> None:
        result = await calc.execute(expression="2 + 3 * 4")
        assert str(result) == "14.0"

    async def test_parentheses_override(self, calc: CalculatorTool) -> None:
        result = await calc.execute(expression="(2 + 3) * 4")
        assert str(result) == "20.0"

    async def test_nested_parentheses(self, calc: CalculatorTool) -> None:
        result = await calc.execute(expression="(2 + (3 * 4)) * 2")
        assert str(result) == "28.0"

    async def test_complex_expression(self, calc: CalculatorTool) -> None:
        result = await calc.execute(expression="(10 + 5) * 3 - 8 / 2")
        assert str(result) == "41.0"


class TestCalculatorEdgeCases:
    """边界与错误情况。"""

    async def test_division_by_zero(self, calc: CalculatorTool) -> None:
        with pytest.raises(ValueError, match="表达式无效"):
            await calc.execute(expression="10 / 0")

    async def test_empty_expression(self, calc: CalculatorTool) -> None:
        with pytest.raises(ValueError, match="不能为空"):
            await calc.execute(expression="")

    async def test_whitespace_only(self, calc: CalculatorTool) -> None:
        with pytest.raises(ValueError, match="不能为空"):
            await calc.execute(expression="   ")

    async def test_invalid_syntax(self, calc: CalculatorTool) -> None:
        with pytest.raises(ValueError, match="表达式无效"):
            await calc.execute(expression="hello world")

    async def test_banned_function_call(self, calc: CalculatorTool) -> None:
        with pytest.raises(ValueError):
            await calc.execute(expression="abs(-5)")

    async def test_float_result(self, calc: CalculatorTool) -> None:
        result = await calc.execute(expression="3.5 * 2")
        assert str(result) == "7.0"

    async def test_large_number(self, calc: CalculatorTool) -> None:
        result = await calc.execute(expression="2 ** 30")
        assert str(result) == "1073741824.0"
