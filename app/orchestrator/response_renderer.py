from __future__ import annotations

from app.orchestrator.types import OrchestratorOutput, ToolExecutionResult


def render_tool_result(tool_result: ToolExecutionResult) -> OrchestratorOutput:
    if not tool_result.ok:
        return OrchestratorOutput(message=f"Tool error ({tool_result.tool_name}): {tool_result.error}")

    return OrchestratorOutput(
        message=(
            f"Tool '{tool_result.tool_name}' executed.\n"
            f"{tool_result.result}"
        )
    )


def render_error(message: str) -> OrchestratorOutput:
    return OrchestratorOutput(message=message)
