from __future__ import annotations

import re
from html import unescape
from typing import Any
from urllib.parse import urlparse

import requests

from app.tools.base import ToolResult


class FetchUrlTool:
    name = "fetch_url"
    description = (
        "Fetch and extract text from a URL. args: {\"url\": \"https://...\", \"max_chars\": 4000}."
    )

    def run(self, args: dict[str, Any]) -> ToolResult:
        url = str(args.get("url", "")).strip()
        if not url:
            return ToolResult(success=False, error="Missing 'url' argument")

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return ToolResult(success=False, error="Only http/https URLs are supported")

        max_chars = args.get("max_chars", 4000)
        try:
            max_chars = max(200, min(int(max_chars), 20000))
        except (TypeError, ValueError):
            return ToolResult(success=False, error="'max_chars' must be an integer")

        try:
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

            # Remove script/style noise first.
            html = re.sub(r"<script[\\s\\S]*?</script>", " ", html, flags=re.IGNORECASE)
            html = re.sub(r"<style[\\s\\S]*?</style>", " ", html, flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", html)
            text = unescape(text)
            text = re.sub(r"\\s+", " ", text).strip()

            return ToolResult(
                success=True,
                data={
                    "url": url,
                    "status": response.status_code,
                    "content_type": response.headers.get("Content-Type", ""),
                    "text": text[:max_chars],
                    "truncated": len(text) > max_chars,
                    "text_length": len(text),
                },
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"Fetch URL failed: {exc}")
