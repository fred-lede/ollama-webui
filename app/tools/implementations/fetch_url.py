from __future__ import annotations

import re
from html import unescape
from typing import Any
from urllib.parse import urlparse

import requests

from app.tools.base import BaseTool


class FetchUrlTool(BaseTool):
    name = "fetch_url"
    description = "Fetch and extract text from a URL. arguments: {\"url\":\"https://...\",\"max_chars\":4000}"

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        url = str(arguments.get("url", "")).strip()
        if not url:
            raise ValueError("Missing 'url' argument")

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Only http/https URLs are supported")

        max_chars = arguments.get("max_chars", 4000)
        try:
            max_chars = max(200, min(int(max_chars), 20000))
        except (TypeError, ValueError):
            raise ValueError("'max_chars' must be an integer")

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        html = response.text

        html = re.sub(r"<script[\\s\\S]*?</script>", " ", html, flags=re.IGNORECASE)
        html = re.sub(r"<style[\\s\\S]*?</style>", " ", html, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", html)
        text = unescape(text)
        text = re.sub(r"\\s+", " ", text).strip()

        return {
            "url": url,
            "status": response.status_code,
            "content_type": response.headers.get("Content-Type", ""),
            "text": text[:max_chars],
            "truncated": len(text) > max_chars,
            "text_length": len(text),
        }
