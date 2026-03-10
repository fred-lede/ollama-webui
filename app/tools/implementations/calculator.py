from __future__ import annotations

import ast
from typing import Any

from app.tools.base import BaseTool


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluate a safe arithmetic expression. arguments: {\"expression\": \"2+2\"}"

    _allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        expression = str(arguments.get("expression", "")).strip()
        if not expression:
            raise ValueError("Missing 'expression' argument")

        node = ast.parse(expression, mode="eval")
        for subnode in ast.walk(node):
            if not isinstance(subnode, self._allowed_nodes):
                raise ValueError("Expression contains unsupported operations")

        value = eval(compile(node, "<calculator>", "eval"), {"__builtins__": {}}, {})
        return {
            "expression": expression,
            "result": value,
        }
