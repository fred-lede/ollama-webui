from __future__ import annotations

import os
from typing import Any

from app.core.app_settings import load_app_settings
from app.tools.base import ToolResult
from app.tools.search_providers.base import SearchProvider
from app.tools.search_providers.serper_provider import SerperProvider
from app.tools.search_providers.tavily_provider import TavilyProvider


class WebSearchTool:
    name = "web_search"
    description = (
        "Search the web. args: {\"query\": \"...\", \"num_results\": 5, "
        "\"provider\": \"serper.dev|tavily\"}."
    )

    def __init__(self, provider: SearchProvider | None = None) -> None:
        self.provider = provider

    def _normalize_provider(self, name: str) -> str:
        normalized = (name or "").strip().lower()
        if normalized in {"serper", "serper.dev", "serper_dev"}:
            return "serper.dev"
        if normalized == "tavily":
            return "tavily"
        return "serper.dev"

    def _build_provider(self, name: str) -> SearchProvider:
        provider_name = self._normalize_provider(name)
        if provider_name == "tavily":
            return TavilyProvider()
        return SerperProvider()

    def run(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(success=False, error="Missing 'query' argument")

        num_results = args.get("num_results", 5)
        try:
            num_results = int(num_results)
        except (TypeError, ValueError):
            return ToolResult(success=False, error="'num_results' must be an integer")

        cfg = load_app_settings().get("search", {})
        default_provider = self._normalize_provider(str(cfg.get("provider", "serper.dev")))
        raw_provider = str(args.get("provider", os.getenv("SEARCH_PROVIDER", default_provider)))
        provider_name = self._normalize_provider(raw_provider)

        try:
            provider = self.provider or self._build_provider(provider_name)
            results = provider.search(query=query, num_results=num_results)
            data = [
                {
                    "title": item.title,
                    "url": item.url,
                    "snippet": item.snippet,
                }
                for item in results
            ]
            return ToolResult(
                success=True,
                data={
                    "provider": provider_name,
                    "query": query,
                    "count": len(data),
                    "results": data,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"Web search failed ({provider_name}): {exc}")
