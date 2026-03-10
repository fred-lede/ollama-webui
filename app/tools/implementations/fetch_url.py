from __future__ import annotations

import re
from html import unescape
from typing import Any
from urllib.parse import urlparse

import requests

from app.core.cancellation import ensure_not_stopped
from app.tools.base import BaseTool


class FetchUrlTool(BaseTool):
    name = "fetch_url"
    description = "Fetch and extract text from a URL. arguments: {\"url\":\"https://...\",\"max_chars\":4000}"

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        ensure_not_stopped()
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
        ensure_not_stopped()
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        ensure_not_stopped()
        # Improve decoding for sites that do not return a reliable charset.
        if not response.encoding or response.encoding.lower() == "iso-8859-1":
            apparent = (response.apparent_encoding or "").strip()
            if apparent:
                response.encoding = apparent
        html = response.text

        # Extract basic metadata before removing tags.
        title_match = re.search(r"<title[^>]*>([\s\S]*?)</title>", html, flags=re.IGNORECASE)
        raw_title = title_match.group(1).strip() if title_match else ""
        title = re.sub(r"\s+", " ", unescape(raw_title)).strip()

        desc_match = re.search(
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([\s\S]*?)["\']',
            html,
            flags=re.IGNORECASE,
        )
        raw_description = desc_match.group(1).strip() if desc_match else ""
        description = re.sub(r"\s+", " ", unescape(raw_description)).strip()

        # Remove non-content blocks before stripping tags.
        html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
        html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
        html = re.sub(r"<noscript[\s\S]*?</noscript>", " ", html, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", html)
        text = unescape(text)
        text = re.sub(r"\s+", " ", text).strip()

        return {
            "url": url,
            "status": response.status_code,
            "content_type": response.headers.get("Content-Type", ""),
            "title": title,
            "description": description,
            "text": text[:max_chars],
            "truncated": len(text) > max_chars,
            "text_length": len(text),
        }
