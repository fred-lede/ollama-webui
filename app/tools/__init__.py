from __future__ import annotations

from app.tools.implementations.calculator import CalculatorTool
from app.tools.implementations.datetime_tool import DateTimeTool
from app.tools.implementations.fetch_url import FetchUrlTool
from app.tools.implementations.web_search import WebSearchTool
from app.tools.registry import ToolRegistry


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(DateTimeTool())
    registry.register(WebSearchTool())
    registry.register(FetchUrlTool())
    return registry
