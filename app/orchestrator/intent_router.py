from __future__ import annotations

import re

from app.orchestrator.types import ToolCall


class ToolIntentRouter:
    """Deterministic first-pass routing before model-based tool planning."""

    def route(self, user_text: str) -> ToolCall | None:
        text = (user_text or "").strip()
        if not text:
            return None

        lowered = text.lower()

        if self._looks_like_time_question(text):
            arguments: dict[str, object] = {}
            timezone = self._infer_timezone(text)
            if timezone:
                arguments["timezone"] = timezone
            return ToolCall(name="datetime", arguments=arguments)

        if any(k in lowered for k in ("calculate", "calc", "幫我算", "計算", "算一下")) or bool(
            re.fullmatch(r"[\d\s+\-*/().%^]+", text)
        ):
            expr = self._extract_expression(text)
            if expr:
                return ToolCall(name="calculator", arguments={"expression": expr})

        if "http://" in lowered or "https://" in lowered:
            if any(k in lowered for k in ("摘要", "summary", "summarize", "抓取", "fetch")):
                match = re.search(r"https?://[^\s)>\"]+", text)
                if match:
                    return ToolCall(name="fetch_url", arguments={"url": match.group(0), "max_chars": 2500})

        if any(k in lowered for k in ("搜尋", "search", "look up", "news", "latest", "find", "查詢", "查一下", "幫我查")):
            query = self._extract_search_query(text)
            return ToolCall(name="web_search", arguments={"query": query})

        return None

    @staticmethod
    def _extract_expression(text: str) -> str | None:
        expr = text.replace("^", "**")
        expr = re.sub(r"[^0-9+\-*/().%\s*]", "", expr)
        expr = re.sub(r"\s+", "", expr)
        return expr if expr else None

    @staticmethod
    def _extract_search_query(text: str) -> str:
        query = text
        for token in ("幫我搜尋", "幫我查", "搜尋", "查詢", "查一下", "search for", "search", "look up", "find"):
            query = query.replace(token, " ")
        query = re.sub(r"\s+", " ", query).strip()
        return query or text

    @staticmethod
    def _looks_like_time_question(text: str) -> bool:
        lowered = text.lower()
        if "time complexity" in lowered:
            return False

        zh_signals = ("現在時間", "現在幾點", "幾點了", "目前時間", "今天日期", "今天幾號", "現在幾月幾號")
        if any(token in text for token in zh_signals):
            return True

        en_signals = (
            "what time is it",
            "current time",
            "time now",
            "what's the time",
            "today date",
            "what date is it",
        )
        return any(token in lowered for token in en_signals)

    @staticmethod
    def _infer_timezone(text: str) -> str | None:
        lowered = text.lower()
        if any(token in lowered for token in ("taipei", "taiwan", "台北", "台灣")):
            return "Asia/Taipei"
        if "utc" in lowered:
            return "UTC"
        if any(token in lowered for token in ("tokyo", "japan", "東京", "日本")):
            return "Asia/Tokyo"
        return None
