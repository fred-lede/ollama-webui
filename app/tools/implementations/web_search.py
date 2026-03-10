from __future__ import annotations

import os
from typing import Any

from app.core.app_settings import load_app_settings
from app.core.cancellation import ensure_not_stopped
from app.tools.base import BaseTool
from app.tools.search_providers.base import SearchProvider
from app.tools.search_providers.serper_provider import SerperProvider
from app.tools.search_providers.tavily_provider import TavilyProvider


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web. arguments: {\"query\":\"...\",\"num_results\":5,\"provider\":\"serper.dev|tavily\"}"

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

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        ensure_not_stopped()
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ValueError("Missing 'query' argument")

        num_results = arguments.get("num_results", 5)
        try:
            num_results = int(num_results)
        except (TypeError, ValueError):
            raise ValueError("'num_results' must be an integer")
        num_results = max(1, min(num_results, 20))

        cfg = load_app_settings().get("search", {})
        default_provider = self._normalize_provider(str(cfg.get("provider", "serper.dev")))
        raw_provider = str(arguments.get("provider", os.getenv("SEARCH_PROVIDER", default_provider)))
        provider_name = self._normalize_provider(raw_provider)

        provider = self.provider or self._build_provider(provider_name)
        ensure_not_stopped()
        results = provider.search(query=query, num_results=num_results)
        ensure_not_stopped()
        data = [
            {
                "title": item.title,
                "url": item.url,
                "snippet": item.snippet,
            }
            for item in results
        ]

        return {
            "provider": provider_name,
            "query": query,
            "count": len(data),
            "results": data,
        }
