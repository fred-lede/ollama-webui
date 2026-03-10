from __future__ import annotations

import re
from html import unescape
from urllib.parse import urlparse

from app.orchestrator.intent_router import ToolIntentRouter
from app.orchestrator.tool_runtime import ToolRuntime
from app.orchestrator.types import AutoToolAction, DirectAction, FetchForLlmAction, SearchForLlmAction

_TRUSTED_SEARCH_DOMAINS = (
    "ollama.com",
    "github.com",
    "docs.ollama.com",
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "wsj.com",
    "nytimes.com",
    "theverge.com",
    "techcrunch.com",
    "arstechnica.com",
    "wired.com",
    "zdnet.com",
)


def normalize_summary_length(value: str | None) -> str:
    v = (value or "").strip().lower()
    if v in {"short", "medium", "long"}:
        return v
    return "medium"


def summary_style_config(summary_length: str) -> dict[str, int | str]:
    level = normalize_summary_length(summary_length)
    if level == "short":
        return {
            "level": "short",
            "snippet_max_chars": 120,
            "points_min": 2,
            "points_max": 3,
            "prompt_hint": "Keep it brief.",
        }
    if level == "long":
        return {
            "level": "long",
            "snippet_max_chars": 360,
            "points_min": 5,
            "points_max": 8,
            "prompt_hint": "Provide a more detailed summary.",
        }
    return {
        "level": "medium",
        "snippet_max_chars": 220,
        "points_min": 3,
        "points_max": 5,
        "prompt_hint": "Keep a balanced level of detail.",
    }


def clean_search_snippet(text: str, max_len: int = 220) -> str:
    snippet = unescape((text or "").strip())
    if not snippet:
        return ""

    snippet = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", snippet)
    snippet = re.sub(r"\]\((https?://[^)]+)\)", " ", snippet)
    snippet = re.sub(r"https?://\S+", " ", snippet)
    snippet = re.sub(r"/Users/\S+", " ", snippet)
    snippet = re.sub(r"\{[^{}]{0,200}\}", " ", snippet)
    snippet = re.sub(r"\|[\s\-:|]+\|", " ", snippet)
    snippet = re.sub(r"(?:\|\s*){2,}", " ", snippet)
    snippet = re.sub(r"\s+", " ", snippet).strip()
    snippet = snippet.strip(" -|:")

    first_segment = re.split(r"[。！？!?]", snippet)[0].strip()
    if len(first_segment) >= 30:
        snippet = first_segment

    if len(snippet) <= max_len:
        return snippet
    return snippet[: max_len - 3].rstrip() + "..."


def summarize_fetched_content(payload: dict[str, object]) -> str:
    title = str(payload.get("title", "")).strip()
    description = str(payload.get("description", "")).strip()
    text = str(payload.get("text", "")).strip()

    cleaned_desc = clean_search_snippet(description, max_len=220) if description else ""
    if cleaned_desc:
        normalized_title = re.sub(r"\s+", "", title.lower())
        normalized_desc = re.sub(r"\s+", "", cleaned_desc.lower())
        generic_tokens = ("資訊系統", "官方網站", "首頁", "welcome", "portal")
        looks_generic = (
            len(cleaned_desc) < 24
            or normalized_desc == normalized_title
            or normalized_desc in normalized_title
            or normalized_title in normalized_desc
            or any(token in cleaned_desc for token in generic_tokens)
        )
        if not looks_generic:
            return cleaned_desc

    chunks = re.split(r"[。！？!?；;\n]", text)
    picked: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        s = clean_search_snippet(chunk, max_len=140)
        if len(s) < 16:
            continue
        if s.lower() in {"english", "中文"}:
            continue
        if s in seen:
            continue
        if any(noisy in s for noisy in (":::", "More...", "cookie", "javascript")):
            continue
        seen.add(s)
        picked.append(s)
        if len(picked) >= 4:
            break

    if picked:
        return "；".join(picked)[:320]
    if cleaned_desc:
        return cleaned_desc
    if title:
        return f"此頁標題為「{title}」，內容以網站導覽或清單為主。"
    return clean_search_snippet(text, max_len=220) if text else "此頁未擷取到可摘要的文字內容。"


def _domain_from_url(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower().strip()
    except Exception:  # noqa: BLE001
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _is_trusted_domain(url: str) -> bool:
    host = _domain_from_url(url)
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in _TRUSTED_SEARCH_DOMAINS)


def _domain_priority(url: str) -> int:
    host = _domain_from_url(url)
    if not host:
        return 99
    if host == "ollama.com" or host.endswith(".ollama.com"):
        return 0
    if host == "github.com" or host.endswith(".github.com"):
        return 1
    return 2


def format_search_results_text(
    provider: str,
    trusted_results_used: bool,
    results: list[dict[str, object]],
    num_results: int,
    summary_length: str,
) -> str:
    cfg = summary_style_config(summary_length)
    snippet_max = int(cfg["snippet_max_chars"])
    lines = [f"搜尋結果（{provider}）："]
    if trusted_results_used:
        lines.append("已啟用可信網域白名單過濾。")
    else:
        lines.append("白名單無匹配，已顯示一般結果。")
    for idx, item in enumerate(results[:num_results], start=1):
        title = str(item.get("title", "(no title)")).strip()
        snippet = clean_search_snippet(str(item.get("snippet", "")), max_len=snippet_max)
        result_url = str(item.get("url", "")).strip()
        lines.append(f"{idx}. {title}")
        if snippet:
            lines.append(f"   {snippet}")
        if result_url:
            lines.append(f"   {result_url}")
    return "\n".join(lines)


class AutoToolPlanner:
    def __init__(self, router: ToolIntentRouter, runtime: ToolRuntime) -> None:
        self.router = router
        self.runtime = runtime

    def plan(
        self,
        text: str | None,
        search_provider: str | None,
        search_num_results: int | float | None,
        search_summary_length: str | None,
    ) -> AutoToolAction | None:
        content = (text or "").strip()
        if not content:
            return None

        tool_call = self.router.route(content)
        if tool_call is None:
            return None

        arguments = dict(tool_call.arguments)
        if tool_call.name == "web_search":
            try:
                num_results = int(search_num_results or 5)
            except (TypeError, ValueError):
                num_results = 5
            arguments["num_results"] = max(1, min(num_results, 20))
            arguments["provider"] = search_provider or "serper.dev"

        tool_result = self.runtime.execute(tool_call.name, arguments)
        if not tool_result.ok:
            return DirectAction(content=f"{tool_call.name} 失敗：{tool_result.error or 'Unknown error'}")

        payload = tool_result.result if isinstance(tool_result.result, dict) else {}

        if tool_call.name == "datetime":
            formatted = str(payload.get("formatted", "")).strip()
            timezone = str(payload.get("timezone", "local")).strip()
            iso = str(payload.get("iso", "")).strip()
            if formatted:
                lines = [f"目前時間：{formatted}", f"時區：{timezone}"]
                if iso:
                    lines.append(f"ISO：{iso}")
                return DirectAction(content="\n".join(lines))
            return DirectAction(content="時間工具已執行，但未回傳可用內容。")

        if tool_call.name == "calculator":
            expression = str(payload.get("expression", arguments.get("expression", ""))).strip()
            return DirectAction(content=f"計算結果：{expression} = {payload.get('result')}")

        if tool_call.name == "fetch_url":
            url = str(payload.get("url", arguments.get("url", ""))).strip()
            return FetchForLlmAction(
                url=url,
                status=payload.get("status", ""),
                title=str(payload.get("title", "")).strip(),
                description=str(payload.get("description", "")).strip(),
                text=str(payload.get("text", "")).strip(),
                fallback_summary=summarize_fetched_content(payload),
            )

        if tool_call.name == "web_search":
            all_results = payload.get("results", []) if isinstance(payload.get("results", []), list) else []
            trusted_results = [item for item in all_results if _is_trusted_domain(str(item.get("url", "")))]
            if trusted_results:
                trusted_results.sort(key=lambda item: _domain_priority(str(item.get("url", ""))))
            results = trusted_results if trusted_results else all_results
            if not results:
                return DirectAction(content="搜尋完成，但沒有找到結果。")

            summary_length = normalize_summary_length(search_summary_length)
            num_results = int(arguments.get("num_results", 5))
            provider_name = str(payload.get("provider", search_provider or "serper.dev"))
            query = str(payload.get("query", arguments.get("query", content))).strip()
            fallback_content = format_search_results_text(
                provider=provider_name,
                trusted_results_used=bool(trusted_results),
                results=results,
                num_results=num_results,
                summary_length=summary_length,
            )
            return SearchForLlmAction(
                provider=provider_name,
                query=query,
                summary_length=summary_length,
                trusted_results_used=bool(trusted_results),
                results=results[:num_results],
                fallback_content=fallback_content,
            )

        return None
