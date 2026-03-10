from __future__ import annotations

import ast
from typing import Any

from app.tools.base import ToolResult


class CalculatorTool:
    name = "calculator"
    description = "Evaluate a safe arithmetic expression. args: {\"expression\": \"2+2\"}"

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

    def run(self, args: dict[str, Any]) -> ToolResult:
        expression = str(args.get("expression", "")).strip()
        if not expression:
            return ToolResult(success=False, error="Missing 'expression' argument")

        try:
            node = ast.parse(expression, mode="eval")
            for subnode in ast.walk(node):
                if not isinstance(subnode, self._allowed_nodes):
                    return ToolResult(success=False, error="Expression contains unsupported operations")

            value = eval(compile(node, "<calculator>", "eval"), {"__builtins__": {}}, {})
            return ToolResult(success=True, data={"expression": expression, "result": value})
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"Calculation failed: {exc}")
