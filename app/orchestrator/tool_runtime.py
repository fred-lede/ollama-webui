from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.cancellation import ensure_not_stopped
from app.orchestrator.policy import ToolPolicy
from app.orchestrator.types import ToolExecutionResult
from app.tools.registry import ToolRegistry


@dataclass(slots=True)
class ToolRuntime:
    registry: ToolRegistry
    policy: ToolPolicy

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        ensure_not_stopped()

        if not self.policy.is_tool_allowed(tool_name):
            return ToolExecutionResult(
                ok=False,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' is disabled by policy",
            )

        raw = self.registry.execute(tool_name, arguments)
        if not raw.get("ok"):
            return ToolExecutionResult(
                ok=False,
                tool_name=tool_name,
                error=str(raw.get("error", "Unknown tool error")),
            )

        result_payload = raw.get("result", {}) if isinstance(raw.get("result"), dict) else {}
        return ToolExecutionResult(ok=True, tool_name=tool_name, result=result_payload)
