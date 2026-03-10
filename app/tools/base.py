from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Minimal, consistent interface for all tools."""

    name: str
    description: str

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> Any:
        """Execute tool logic and return structured data."""

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
        }
