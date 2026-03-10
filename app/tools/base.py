from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: str | None = None


class Tool(Protocol):
    name: str
    description: str

    def run(self, args: dict[str, Any]) -> ToolResult:
        ...
