from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepKind(str, Enum):
    ROUTE = "route"
    TOOL = "tool"
    MODEL = "model"
    RENDER = "render"


@dataclass(slots=True)
class OrchestratorStep:
    kind: StepKind
    name: str
    detail: str = ""


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolExecutionResult:
    ok: bool
    tool_name: str
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(slots=True)
class OrchestratorOutput:
    message: str
    sources: list[dict[str, str]] = field(default_factory=list)
    steps: list[OrchestratorStep] = field(default_factory=list)


@dataclass(slots=True)
class DirectAction:
    content: str


@dataclass(slots=True)
class FetchForLlmAction:
    url: str
    status: object
    title: str
    description: str
    text: str
    fallback_summary: str


@dataclass(slots=True)
class SearchForLlmAction:
    provider: str
    query: str
    summary_length: str
    trusted_results_used: bool
    results: list[dict[str, object]] = field(default_factory=list)
    fallback_content: str = ""


AutoToolAction = DirectAction | FetchForLlmAction | SearchForLlmAction
