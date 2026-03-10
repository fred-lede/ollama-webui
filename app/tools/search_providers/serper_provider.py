from __future__ import annotations

import os

import requests

from app.core.app_settings import load_app_settings
from app.tools.search_providers.base import SearchProvider, SearchResult


class SerperProvider(SearchProvider):
    def __init__(self, api_key: str | None = None, timeout: int = 20) -> None:
        settings = load_app_settings().get("search", {})
        self.api_key = api_key or os.getenv("SERPER_API_KEY", "") or str(settings.get("serper_api_key", ""))
        self.timeout = timeout
        self.endpoint = (
            os.getenv("SERPER_API_URL", "")
            or str(settings.get("serper_api_url", "")).strip()
            or "https://google.serper.dev/search"
        )

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        if not self.api_key:
            raise ValueError("SERPER_API_KEY is not set")

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "q": query,
            "num": max(1, min(int(num_results), 10)),
        }

        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()

        body = response.json()
        organic = body.get("organic", [])

        results: list[SearchResult] = []
        for item in organic:
            title = str(item.get("title", "")).strip()
            url = str(item.get("link", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            if not url:
                continue
            results.append(SearchResult(title=title, url=url, snippet=snippet))

        return results
