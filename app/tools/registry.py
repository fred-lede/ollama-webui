from __future__ import annotations

from typing import Any

from app.tools.base import BaseTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, str]]:
        return [{"name": tool.name, "description": tool.description} for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self.get(name)
        if tool is None:
            return {
                "ok": False,
                "tool_name": name,
                "error": f"Tool '{name}' not found",
            }

        try:
            result = tool.run(arguments)
            return {
                "ok": True,
                "tool_name": name,
                "result": result,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "tool_name": name,
                "error": f"Tool execution failed: {exc}",
            }
