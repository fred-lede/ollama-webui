from __future__ import annotations

import json
import re
from typing import Any

from app.tools.registry import ToolRegistry

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class ToolRouter:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def extract_tool_call(self, text: str) -> dict[str, Any] | None:
        match = _TOOL_CALL_PATTERN.search(text or "")
        if not match:
            return None

        raw_json = match.group(1).strip()
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            return None
        if not isinstance(arguments, dict):
            return None

        return {
            "name": name.strip(),
            "arguments": arguments,
        }

    def execute_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        name = str(tool_call.get("name", "")).strip()
        arguments = tool_call.get("arguments", {})

        if not name:
            return {
                "ok": False,
                "tool_name": "",
                "error": "Invalid tool call: missing name",
            }

        if not isinstance(arguments, dict):
            return {
                "ok": False,
                "tool_name": name,
                "error": "Invalid tool call: arguments must be an object",
            }

        try:
            tool = self.registry.get(name)
            if tool is None:
                return {
                    "ok": False,
                    "tool_name": name,
                    "error": f"Tool '{name}' not found",
                }

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

    def remove_tool_call_markup(self, text: str) -> str:
        return _TOOL_CALL_PATTERN.sub("", text or "").strip()
