from __future__ import annotations

from dataclasses import dataclass

from app.orchestrator.context import RequestContext
from app.orchestrator.intent_router import ToolIntentRouter
from app.orchestrator.response_renderer import render_tool_result
from app.orchestrator.tool_runtime import ToolRuntime
from app.orchestrator.types import OrchestratorOutput, OrchestratorStep, StepKind


@dataclass(slots=True)
class Orchestrator:
    runtime: ToolRuntime
    router: ToolIntentRouter

    def process(self, ctx: RequestContext, user_text: str) -> OrchestratorOutput:
        steps: list[OrchestratorStep] = [OrchestratorStep(kind=StepKind.ROUTE, name="deterministic-route")]

        tool_call = self.router.route(user_text)
        if tool_call is None:
            return OrchestratorOutput(
                message="No deterministic tool route matched. Fall through to model pipeline.",
                steps=steps,
            )

        steps.append(OrchestratorStep(kind=StepKind.TOOL, name=tool_call.name))
        result = self.runtime.execute(tool_call.name, tool_call.arguments)
        out = render_tool_result(result)
        out.steps = steps
        return out
