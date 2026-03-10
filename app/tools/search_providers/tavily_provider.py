from __future__ import annotations

import os

import requests

from app.core.app_settings import load_app_settings
from app.core.cancellation import ensure_not_stopped
from app.tools.search_providers.base import SearchProvider, SearchResult


class TavilyProvider(SearchProvider):
    def __init__(self, api_key: str | None = None, timeout: int = 20) -> None:
        settings = load_app_settings().get("search", {})
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "") or str(settings.get("tavily_api_key", ""))
        self.timeout = timeout
        self.endpoint = (
            os.getenv("TAVILY_API_URL", "")
            or str(settings.get("tavily_api_url", "")).strip()
            or "https://api.tavily.com/search"
        )

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        ensure_not_stopped()
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is not set")

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max(1, min(int(num_results), 20)),
            "include_answer": False,
            "include_raw_content": False,
        }

        response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()
        ensure_not_stopped()

        body = response.json()
        items = body.get("results", [])

        results: list[SearchResult] = []
        for item in items:
            ensure_not_stopped()
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            snippet = str(item.get("content", "")).strip()
            if not url:
                continue
            results.append(SearchResult(title=title, url=url, snippet=snippet))

        return results
