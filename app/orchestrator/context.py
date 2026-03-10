from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelOptions:
    temperature: float | None = None
    num_predict: int | None = None
    top_p: float | None = None
    typical_p: float | None = None
    num_ctx: int | None = None

    def to_ollama_options(self) -> dict[str, float | int]:
        options: dict[str, float | int] = {}
        if self.temperature is not None:
            options["temperature"] = float(self.temperature)
        if self.num_predict is not None:
            options["num_predict"] = int(self.num_predict)
        if self.top_p is not None:
            options["top_p"] = float(self.top_p)
        if self.typical_p is not None:
            options["typical_p"] = float(self.typical_p)
        if self.num_ctx is not None:
            options["num_ctx"] = int(self.num_ctx)
        return options


@dataclass(slots=True)
class SearchSettings:
    provider: str = "serper.dev"
    num_results: int = 5
    summary_length: str = "medium"


@dataclass(slots=True)
class RequestContext:
    trace_id: str
    model: str
    server_url: str
    options: ModelOptions
    search: SearchSettings
    metadata: dict[str, Any] = field(default_factory=dict)
