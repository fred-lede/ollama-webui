from __future__ import annotations

import base64
import json
import logging
import re
import time
from html import unescape
from urllib.parse import urlparse

import gradio as gr
import requests

from app.core.cancellation import (
    OperationCancelled,
    clear_stop,
    ensure_not_stopped,
    is_stop_requested,
    request_stop,
)
from app.core.tool_router import ToolRouter
from app.tools import build_default_registry

tool_registry = build_default_registry()
tool_router = ToolRouter(tool_registry)

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "for",
    "in",
    "on",
    "with",
    "is",
    "are",
    "be",
    "this",
    "that",
    "please",
    "what",
    "how",
    "why",
    "when",
    "where",
}

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


def _normalize_summary_length(value: str | None) -> str:
    v = (value or "").strip().lower()
    if v in {"short", "medium", "long"}:
        return v
    return "medium"


def _summary_style_config(summary_length: str) -> dict[str, int | str]:
    level = _normalize_summary_length(summary_length)
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


def _extract_search_keywords(text: str) -> str:
    cleaned = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text.lower())
    tokens = re.findall(r"[a-z0-9][a-z0-9._-]*|[\u4e00-\u9fff]{2,}", cleaned)

    keywords: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in _STOPWORDS:
            continue
        if len(token) <= 1:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= 8:
            break

    return " ".join(keywords) if keywords else text.strip()


def _build_tool_instruction() -> str:
    lines = ["Available tools:"]
    for tool in tool_registry.list_tools():
        lines.append(f"- {tool['name']}: {tool['description']}")
    lines.append("If a tool is needed, respond ONLY with:")
    lines.append('<tool_call>{"name":"tool_name","arguments":{}}</tool_call>')
    lines.append("Do not include extra text when emitting a tool call.")
    return "\n".join(lines)


def _linkify_reference_markers(text: str, refs: list[dict[str, str]]) -> str:
    if not text or not refs:
        return text

    def repl(match: re.Match[str]) -> str:
        idx = int(match.group(1)) - 1
        if idx < 0 or idx >= len(refs):
            return match.group(0)
        url = refs[idx].get("url", "").strip()
        if not url:
            return match.group(0)
        return f"[{idx + 1}]({url})"

    return re.sub(r"\[(\d+)\]", repl, text)


def _append_source_links(text: str, refs: list[dict[str, str]]) -> str:
    if not text or not refs:
        return text

    source_lines: list[str] = []
    for idx, ref in enumerate(refs, start=1):
        url = str(ref.get("url", "")).strip()
        title = str(ref.get("title", "")).strip() or url
        if not url:
            continue
        source_lines.append(f"{idx}. [{title}]({url})")

    if not source_lines:
        return text

    if "Web Sources:" in text:
        return text

    return text + "\n\nWeb Sources:\n" + "\n".join(source_lines)


def _build_web_context(
    question_text: str,
    search_provider: str | None,
    search_num_results: int | float | None,
) -> tuple[str | None, str | None, list[dict[str, str]]]:
    ensure_not_stopped()
    keywords = _extract_search_keywords(question_text)
    try:
        num_results = int(search_num_results or 5)
    except (TypeError, ValueError):
        num_results = 5
    num_results = max(1, min(num_results, 20))

    result = tool_registry.execute(
        "web_search",
        {
            "query": keywords,
            "num_results": num_results,
            "provider": search_provider or "serper.dev",
        },
    )
    ensure_not_stopped()

    if not result.get("ok"):
        return None, str(result.get("error", "Unknown search error")), []

    payload = result.get("result", {}) if isinstance(result.get("result"), dict) else {}
    results = payload.get("results", []) if isinstance(payload.get("results", []), list) else []

    if not results:
        return None, "No search results", []

    refs: list[dict[str, str]] = []
    lines = [
        "Web Search Digest (for up-to-date context):",
        f"Search Provider: {payload.get('provider', search_provider or 'serper.dev')}",
        f"Search Keywords: {keywords}",
    ]

    for idx, item in enumerate(results[:num_results], start=1):
        title = str(item.get("title", "(no title)")).strip()
        snippet = str(item.get("snippet", "")).strip()
        url = str(item.get("url", "")).strip()

        refs.append({"title": title, "url": url})

        lines.append(f"[{idx}] Title: {title}")
        if snippet:
            lines.append(f"[{idx}] Summary: {snippet}")
        if url:
            lines.append(f"[{idx}] Source: {url}")

    return "\n".join(lines), None, refs


def _try_execute_tool_command(text: str | None, search_provider: str | None = None) -> str | None:
    ensure_not_stopped()
    if not text:
        return None

    content = text.strip()
    if not content:
        return None

    if content == "/tools":
        tools = tool_registry.list_tools()
        if not tools:
            return "No tools are registered."
        lines = ["Available tools:"]
        for item in tools:
            lines.append(f"- {item['name']}: {item['description']}")
        return "\n".join(lines)

    if not content.startswith("/tool "):
        return None

    command = content[len("/tool ") :].strip()
    if not command:
        return "Usage: /tool <name> <json-args>"

    tool_name, _, arg_str = command.partition(" ")
    arguments: dict[str, object] = {}

    if arg_str.strip():
        try:
            parsed = json.loads(arg_str)
            if not isinstance(parsed, dict):
                return "Tool args must be a JSON object."
            arguments = parsed
        except json.JSONDecodeError as exc:
            return f"Invalid JSON args: {exc.msg}"

    if tool_name == "web_search" and search_provider:
        arguments.setdefault("provider", search_provider)

    result = tool_registry.execute(tool_name, arguments)
    if not result.get("ok"):
        return f"Tool '{tool_name}' failed: {result.get('error')}"

    return f"Tool '{tool_name}' result:\n{json.dumps(result.get('result', {}), ensure_ascii=False, indent=2)}"


def _extract_first_url(text: str) -> str | None:
    match = re.search(r"https?://[^\s)>\"]+", text)
    return match.group(0) if match else None


def _clean_search_snippet(text: str, max_len: int = 220) -> str:
    snippet = unescape((text or "").strip())
    if not snippet:
        return ""

    # Drop markdown links and dangling markdown/url artifacts.
    snippet = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", snippet)
    snippet = re.sub(r"\]\((https?://[^)]+)\)", " ", snippet)
    snippet = re.sub(r"https?://\S+", " ", snippet)
    snippet = re.sub(r"/Users/\S+", " ", snippet)

    # Remove JSON-like structured noise that often appears in raw crawl results.
    snippet = re.sub(r"\{[^{}]{0,200}\}", " ", snippet)
    # Remove markdown-like table fragments.
    snippet = re.sub(r"\|[\s\-:|]+\|", " ", snippet)
    snippet = re.sub(r"(?:\|\s*){2,}", " ", snippet)

    snippet = re.sub(r"\s+", " ", snippet).strip()
    snippet = snippet.strip(" -|:")

    # Keep only the first readable sentence-like segment to reduce crawl noise.
    first_segment = re.split(r"[。！？!?]", snippet)[0].strip()
    if len(first_segment) >= 30:
        snippet = first_segment

    if len(snippet) <= max_len:
        return snippet
    return snippet[: max_len - 3].rstrip() + "..."


def _summarize_fetched_content(payload: dict[str, object]) -> str:
    title = str(payload.get("title", "")).strip()
    description = str(payload.get("description", "")).strip()
    text = str(payload.get("text", "")).strip()

    cleaned_desc = _clean_search_snippet(description, max_len=220) if description else ""
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

    # Fallback local summary when LLM summarization is unavailable.
    chunks = re.split(r"[。！？!?；;\n]", text)
    picked: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        s = _clean_search_snippet(chunk, max_len=140)
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

    return _clean_search_snippet(text, max_len=220) if text else "此頁未擷取到可摘要的文字內容。"


def _format_search_results_text(
    provider: str,
    trusted_results_used: bool,
    results: list[dict[str, object]],
    num_results: int,
    summary_length: str,
) -> str:
    cfg = _summary_style_config(summary_length)
    snippet_max = int(cfg["snippet_max_chars"])
    lines = [f"搜尋結果（{provider}）："]
    if trusted_results_used:
        lines.append("已啟用可信網域白名單過濾。")
    else:
        lines.append("白名單無匹配，已顯示一般結果。")
    for idx, item in enumerate(results[:num_results], start=1):
        title = str(item.get("title", "(no title)")).strip()
        snippet = _clean_search_snippet(str(item.get("snippet", "")), max_len=snippet_max)
        result_url = str(item.get("url", "")).strip()
        lines.append(f"{idx}. {title}")
        if snippet:
            lines.append(f"   {snippet}")
        if result_url:
            lines.append(f"   {result_url}")
    return "\n".join(lines)


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


def _looks_like_calculation_request(text: str) -> bool:
    lowered = text.lower()
    if any(token in lowered for token in ("calculate", "calc", "計算", "算一下", "幫我算")):
        return True
    return bool(re.fullmatch(r"[\d\s+\-*/().%^]+", text.strip()))


def _extract_expression(text: str) -> str | None:
    candidates = [
        text,
        text.replace("請幫我算", "").replace("幫我算", "").replace("計算", "").replace("calculate", ""),
    ]
    for candidate in candidates:
        expr = candidate.strip()
        expr = expr.replace("^", "**")
        expr = re.sub(r"[^0-9+\-*/().%\s*]", "", expr)
        expr = re.sub(r"\s+", "", expr)
        if expr and re.fullmatch(r"[0-9+\-*/().%*]+", expr):
            return expr
    return None


def _looks_like_fetch_url_request(text: str) -> bool:
    lowered = text.lower()
    if not _extract_first_url(text):
        return False
    fetch_signals = ("摘要", "重點", "內容", "網頁", "網址", "讀取", "抓取", "fetch", "summarize", "summary")
    return any(token in lowered for token in fetch_signals)


def _looks_like_web_search_request(text: str) -> bool:
    lowered = text.lower()
    if _extract_first_url(text):
        return False
    signals = (
        "幫我查",
        "幫我搜尋",
        "搜尋",
        "查詢",
        "查一下",
        "search",
        "look up",
        "find",
        "latest",
        "news",
    )
    return any(token in lowered for token in signals)


def _infer_timezone_from_text(text: str) -> str | None:
    lowered = text.lower()
    if any(token in lowered for token in ("taipei", "taiwan", "台北", "台灣")):
        return "Asia/Taipei"
    if "utc" in lowered:
        return "UTC"
    if any(token in lowered for token in ("tokyo", "japan", "東京", "日本")):
        return "Asia/Tokyo"
    return None


def _try_auto_tool_intent(
    text: str | None,
    search_provider: str | None,
    search_num_results: int | float | None,
    search_summary_length: str | None,
) -> dict[str, object] | None:
    content = (text or "").strip()
    if not content:
        return None

    if not _looks_like_time_question(content):
        pass
    else:
        arguments: dict[str, object] = {}
        tz_name = _infer_timezone_from_text(content)
        if tz_name:
            arguments["timezone"] = tz_name

        result = tool_registry.execute("datetime", arguments)
        if result.get("ok"):
            payload = result.get("result", {}) if isinstance(result.get("result"), dict) else {}
            formatted = str(payload.get("formatted", "")).strip()
            timezone = str(payload.get("timezone", "local")).strip()
            iso = str(payload.get("iso", "")).strip()
            if formatted:
                lines = [f"目前時間：{formatted}", f"時區：{timezone}"]
                if iso:
                    lines.append(f"ISO：{iso}")
                return {"mode": "direct", "content": "\n".join(lines)}

    if _looks_like_calculation_request(content):
        expr = _extract_expression(content)
        if expr:
            result = tool_registry.execute("calculator", {"expression": expr})
            if result.get("ok"):
                payload = result.get("result", {}) if isinstance(result.get("result"), dict) else {}
                return {"mode": "direct", "content": f"計算結果：{payload.get('expression', expr)} = {payload.get('result')}"}
            return {"mode": "direct", "content": f"計算失敗：{result.get('error', 'Unknown error')}"}

    if _looks_like_fetch_url_request(content):
        url = _extract_first_url(content)
        if url:
            result = tool_registry.execute("fetch_url", {"url": url, "max_chars": 2500})
            if result.get("ok"):
                payload = result.get("result", {}) if isinstance(result.get("result"), dict) else {}
                return {
                    "mode": "fetch_for_llm",
                    "url": str(payload.get("url", url)).strip(),
                    "status": payload.get("status", ""),
                    "title": str(payload.get("title", "")).strip(),
                    "description": str(payload.get("description", "")).strip(),
                    "text": str(payload.get("text", "")).strip(),
                    "fallback_summary": _summarize_fetched_content(payload),
                }
            return {"mode": "direct", "content": f"抓取網址失敗：{result.get('error', 'Unknown error')}"}

    if _looks_like_web_search_request(content):
        query = content
        for token in ("幫我搜尋", "幫我查", "搜尋", "查詢", "查一下", "search for", "search", "look up", "find"):
            query = query.replace(token, " ")
        query = re.sub(r"\s+", " ", query).strip() or content
        try:
            num_results = int(search_num_results or 5)
        except (TypeError, ValueError):
            num_results = 5
        num_results = max(1, min(num_results, 20))
        summary_length = _normalize_summary_length(search_summary_length)

        result = tool_registry.execute(
            "web_search",
            {
                "query": query,
                "num_results": num_results,
                "provider": search_provider or "serper.dev",
            },
        )
        if result.get("ok"):
            payload = result.get("result", {}) if isinstance(result.get("result"), dict) else {}
            all_results = payload.get("results", []) if isinstance(payload.get("results", []), list) else []
            trusted_results = [item for item in all_results if _is_trusted_domain(str(item.get("url", "")))]
            if trusted_results:
                trusted_results.sort(key=lambda item: _domain_priority(str(item.get("url", ""))))
            results = trusted_results if trusted_results else all_results
            if not results:
                return {"mode": "direct", "content": "搜尋完成，但沒有找到結果。"}
            provider_name = str(payload.get("provider", search_provider or "serper.dev"))
            fallback_content = _format_search_results_text(
                provider=provider_name,
                trusted_results_used=bool(trusted_results),
                results=results,
                num_results=num_results,
                summary_length=summary_length,
            )
            return {
                "mode": "search_for_llm",
                "provider": provider_name,
                "query": query,
                "summary_length": summary_length,
                "trusted_results_used": bool(trusted_results),
                "results": results[:num_results],
                "fallback_content": fallback_content,
            }
        return {"mode": "direct", "content": f"搜尋失敗：{result.get('error', 'Unknown error')}"}

    return None


def _build_history_prompt(history: list[dict[str, str]], user_text: str) -> str:
    prompt = "Previous conversation:\n" if history else ""
    for msg in history:
        prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    prompt += f"\nUser: {user_text}\n"
    return prompt


def _build_ollama_options(
    llm_temperature,
    llm_max_tokens,
    llm_top_p,
    llm_typical_p,
    llm_num_ctx,
) -> dict[str, float | int]:
    options: dict[str, float | int] = {}
    try:
        if llm_temperature is not None:
            options["temperature"] = float(llm_temperature)
    except (TypeError, ValueError):
        pass
    try:
        if llm_max_tokens is not None:
            options["num_predict"] = int(llm_max_tokens)
    except (TypeError, ValueError):
        pass
    try:
        if llm_top_p is not None:
            options["top_p"] = float(llm_top_p)
    except (TypeError, ValueError):
        pass
    try:
        if llm_typical_p is not None:
            options["typical_p"] = float(llm_typical_p)
    except (TypeError, ValueError):
        pass
    try:
        if llm_num_ctx is not None:
            options["num_ctx"] = int(llm_num_ctx)
    except (TypeError, ValueError):
        pass
    return options


def _build_chat_url(selected_server: str) -> str:
    raw = (selected_server or "").strip()
    if not raw:
        raise ValueError("Server is empty")

    if "://" not in raw:
        raw = "http://" + raw

    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"Invalid server address: {selected_server}")

    address = f"{parsed.scheme}://{parsed.hostname}"
    port = parsed.port if parsed.port else 11434
    return f"{address}:{port}/api/chat"


def _post_chat_once(url: str, headers: dict[str, str], payload: dict) -> str:
    ensure_not_stopped()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        body = response.json()
        return str(body.get("message", {}).get("content", ""))
    except requests.RequestException as exc:
        # Some Ollama-compatible servers fail on non-stream mode. Retry with stream mode.
        logging.info("Non-stream chat failed, retrying with stream mode: %s", exc)
        stream_payload = dict(payload)
        stream_payload["stream"] = True

        chunks: list[str] = []
        for chunk in _stream_chat(url, headers, stream_payload):
            ensure_not_stopped()
            if chunk:
                chunks.append(chunk)
        return "".join(chunks)


def _build_fallback_payloads(
    model: str,
    prompt: str,
    user_text: str,
    options: dict[str, float | int],
    image_payload: dict[str, object],
) -> list[tuple[str, dict[str, object]]]:
    payloads: list[tuple[str, dict[str, object]]] = []

    full_message: dict[str, object] = {"role": "user", "content": prompt}
    if image_payload:
        full_message.update(image_payload)

    user_only_message: dict[str, object] = {"role": "user", "content": user_text or prompt}
    if image_payload:
        user_only_message.update(image_payload)

    payloads.append(
        (
            "full_prompt_with_options",
            {
                "model": model,
                "messages": [full_message],
                "options": options,
                "stream": True,
            },
        )
    )
    payloads.append(
        (
            "full_prompt_without_options",
            {
                "model": model,
                "messages": [full_message],
                "stream": True,
            },
        )
    )
    payloads.append(
        (
            "user_only_with_options",
            {
                "model": model,
                "messages": [user_only_message],
                "options": options,
                "stream": True,
            },
        )
    )
    payloads.append(
        (
            "user_only_without_options",
            {
                "model": model,
                "messages": [user_only_message],
                "stream": True,
            },
        )
    )

    return payloads


def _stream_chat(url: str, headers: dict[str, str], payload: dict):
    ensure_not_stopped()
    response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
    response.raise_for_status()
    try:
        for line in response.iter_lines():
            if is_stop_requested():
                response.close()
                raise OperationCancelled("Operation cancelled by user")
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    finally:
        response.close()


def ask_question_stream(
    question,
    history,
    model,
    selected_server,
    llm_temperature,
    llm_max_tokens,
    llm_top_p,
    llm_typical_p,
    llm_num_ctx,
    search_provider,
    web_search_enabled,
    search_num_results,
    search_summary_length,
):
    clear_stop()
    ensure_not_stopped()

    user_text = question.get("text") or ""
    web_refs: list[dict[str, str]] = []

    new_messages = history.copy()
    new_messages.append({"role": "user", "content": user_text})

    assistant_message = {"role": "assistant", "content": ""}
    new_messages.append(assistant_message)

    # 1) Manual tool commands (existing behavior retained)
    tool_output = _try_execute_tool_command(user_text, search_provider=search_provider)
    if tool_output is not None:
        assistant_message["content"] = tool_output
        yield new_messages[-1], "Tool command executed"
        return

    auto_tool_action = _try_auto_tool_intent(
        user_text,
        search_provider=search_provider,
        search_num_results=search_num_results,
        search_summary_length=search_summary_length,
    )
    if auto_tool_action is not None:
        mode = str(auto_tool_action.get("mode", "")).strip()
        if mode == "direct":
            assistant_message["content"] = str(auto_tool_action.get("content", "")).strip() or "Tool executed."
            yield new_messages[-1], "Tool intent executed"
            return

    if model is None:
        if auto_tool_action is not None:
            auto_mode = str(auto_tool_action.get("mode", "")).strip()
            if auto_mode == "fetch_for_llm":
                fetched_url = str(auto_tool_action.get("url", "")).strip()
                fetched_status = auto_tool_action.get("status", "")
                fetched_title = str(auto_tool_action.get("title", "")).strip()
                fallback_summary = str(auto_tool_action.get("fallback_summary", "")).strip()
                lines = [f"已抓取網址：{fetched_url}", f"狀態碼：{fetched_status}"]
                if fetched_title:
                    lines.append(f"標題：{fetched_title}")
                lines.append(f"內容摘要：{fallback_summary}")
                assistant_message["content"] = "\n".join(lines)
                yield new_messages[-1], "Tool intent executed (fallback summary)"
                return
            if auto_mode == "search_for_llm":
                assistant_message["content"] = str(auto_tool_action.get("fallback_content", "")).strip() or "搜尋完成。"
                yield new_messages[-1], "Tool intent executed (fallback summary)"
                return
        assistant_message["content"] = "Please select a model."
        yield new_messages[-1], "Missing model"
        return

    if selected_server is None:
        if auto_tool_action is not None:
            auto_mode = str(auto_tool_action.get("mode", "")).strip()
            if auto_mode == "fetch_for_llm":
                fetched_url = str(auto_tool_action.get("url", "")).strip()
                fetched_status = auto_tool_action.get("status", "")
                fetched_title = str(auto_tool_action.get("title", "")).strip()
                fallback_summary = str(auto_tool_action.get("fallback_summary", "")).strip()
                lines = [f"已抓取網址：{fetched_url}", f"狀態碼：{fetched_status}"]
                if fetched_title:
                    lines.append(f"標題：{fetched_title}")
                lines.append(f"內容摘要：{fallback_summary}")
                assistant_message["content"] = "\n".join(lines)
                yield new_messages[-1], "Tool intent executed (fallback summary)"
                return
            if auto_mode == "search_for_llm":
                assistant_message["content"] = str(auto_tool_action.get("fallback_content", "")).strip() or "搜尋完成。"
                yield new_messages[-1], "Tool intent executed (fallback summary)"
                return
        assistant_message["content"] = "Please select a server."
        yield new_messages[-1], "Missing server"
        return

    try:
        ensure_not_stopped()
        url = _build_chat_url(str(selected_server))
        headers = {"Content-Type": "application/json"}
        options = _build_ollama_options(
            llm_temperature,
            llm_max_tokens,
            llm_top_p,
            llm_typical_p,
            llm_num_ctx,
        )

        prompt = _build_history_prompt(history, user_text)

        if auto_tool_action is not None and str(auto_tool_action.get("mode", "")) == "fetch_for_llm":
            fetched_url = str(auto_tool_action.get("url", "")).strip()
            fetched_status = auto_tool_action.get("status", "")
            fetched_title = str(auto_tool_action.get("title", "")).strip()
            fetched_description = str(auto_tool_action.get("description", "")).strip()
            fetched_text = str(auto_tool_action.get("text", "")).strip()
            fallback_summary = str(auto_tool_action.get("fallback_summary", "")).strip()

            summary_prompt = (
                "You are summarizing a fetched webpage for the user.\n"
                "Return Traditional Chinese.\n"
                "Output format:\n"
                "1) 一句主旨\n"
                "2) 3-5點重點（條列）\n"
                "3) 若頁面可操作，補一行「可做的事」\n"
                "Keep concise and factual. Do not fabricate missing facts.\n\n"
                f"User request: {user_text}\n"
                f"URL: {fetched_url}\n"
                f"HTTP status: {fetched_status}\n"
                f"Page title: {fetched_title}\n"
                f"Meta description: {fetched_description}\n"
                "Fetched content:\n"
                f"{fetched_text[:3500]}\n"
            )

            summary_payload = {
                "model": model,
                "messages": [{"role": "user", "content": summary_prompt}],
                "options": options,
                "stream": True,
            }

            yield new_messages[-1], "Summarizing fetched content"
            try:
                for chunk in _stream_chat(url, headers, summary_payload):
                    ensure_not_stopped()
                    assistant_message["content"] += chunk
                    new_messages[-1]["content"] = assistant_message["content"]
                    yield new_messages[-1], "Summarizing fetched content"
                    time.sleep(0.02)
            except requests.RequestException as exc:
                logging.warning("LLM summary for fetched URL failed: %s", exc)

            if not assistant_message["content"].strip():
                retry_payload = dict(summary_payload)
                retry_payload["stream"] = False
                try:
                    retry_text = _post_chat_once(url, headers, retry_payload).strip()
                    if retry_text:
                        assistant_message["content"] = retry_text
                        new_messages[-1]["content"] = assistant_message["content"]
                except requests.RequestException as exc:
                    logging.warning("Retry LLM summary for fetched URL failed: %s", exc)

            if not assistant_message["content"].strip():
                logging.info("Using fallback summary for fetched URL.")
                lines = [f"已抓取網址：{fetched_url}", f"狀態碼：{fetched_status}"]
                if fetched_title:
                    lines.append(f"標題：{fetched_title}")
                lines.append(f"內容摘要：{fallback_summary}")
                assistant_message["content"] = "\n".join(lines)
                new_messages[-1]["content"] = assistant_message["content"]

            yield new_messages[-1], "Completed"
            return

        if auto_tool_action is not None and str(auto_tool_action.get("mode", "")) == "search_for_llm":
            search_provider_name = str(auto_tool_action.get("provider", search_provider or "serper.dev")).strip()
            search_query = str(auto_tool_action.get("query", user_text)).strip()
            summary_length = _normalize_summary_length(str(auto_tool_action.get("summary_length", search_summary_length)))
            style_cfg = _summary_style_config(summary_length)
            points_min = int(style_cfg["points_min"])
            points_max = int(style_cfg["points_max"])
            prompt_hint = str(style_cfg["prompt_hint"])
            search_results = auto_tool_action.get("results", [])
            fallback_content = str(auto_tool_action.get("fallback_content", "")).strip()

            if isinstance(search_results, list):
                compact_results = [
                    {
                        "title": str(item.get("title", "")).strip(),
                        "snippet": _clean_search_snippet(str(item.get("snippet", "")), max_len=260),
                        "url": str(item.get("url", "")).strip(),
                    }
                    for item in search_results
                ]
            else:
                compact_results = []

            summary_prompt = (
                "You are summarizing web search results for the user.\n"
                "Return Traditional Chinese.\n"
                "Output format:\n"
                "1) 一句結論\n"
                f"2) {points_min}-{points_max}點重點（條列）\n"
                "3) 資料來源（列出編號與網址）\n"
                f"{prompt_hint} Keep concise and factual. Do not fabricate missing facts.\n\n"
                f"User request: {user_text}\n"
                f"Search query: {search_query}\n"
                f"Search provider: {search_provider_name}\n"
                "Search results (JSON):\n"
                + json.dumps(compact_results, ensure_ascii=False, indent=2)
            )

            summary_payload = {
                "model": model,
                "messages": [{"role": "user", "content": summary_prompt}],
                "options": options,
                "stream": True,
            }

            yield new_messages[-1], "Summarizing search results"
            try:
                for chunk in _stream_chat(url, headers, summary_payload):
                    ensure_not_stopped()
                    assistant_message["content"] += chunk
                    new_messages[-1]["content"] = assistant_message["content"]
                    yield new_messages[-1], "Summarizing search results"
                    time.sleep(0.02)
            except requests.RequestException as exc:
                logging.warning("LLM summary for web search failed: %s", exc)

            if not assistant_message["content"].strip():
                retry_payload = dict(summary_payload)
                retry_payload["stream"] = False
                try:
                    retry_text = _post_chat_once(url, headers, retry_payload).strip()
                    if retry_text:
                        assistant_message["content"] = retry_text
                        new_messages[-1]["content"] = assistant_message["content"]
                except requests.RequestException as exc:
                    logging.warning("Retry LLM summary for web search failed: %s", exc)

            if not assistant_message["content"].strip():
                logging.info("Using fallback summary for web search.")
                assistant_message["content"] = fallback_content or "搜尋完成。"
                new_messages[-1]["content"] = assistant_message["content"]

            yield new_messages[-1], "Completed"
            return

        if bool(web_search_enabled) and user_text.strip():
            web_context, web_error, web_refs = _build_web_context(user_text, search_provider, search_num_results)
            if web_context:
                prompt += (
                    "\n\n" + web_context + "\n\n"
                    "Instruction: Base your answer on the web search digest and cite source indices like [1], [2]."
                )
            elif web_error:
                logging.warning("Web search failed while enabled: %s", web_error)
                assistant_message["content"] = (
                    "網路搜尋已開啟，但搜尋失敗：" + str(web_error) + "。\n"
                    "請確認已設定對應 API Key（Serper: SERPER_API_KEY / Tavily: TAVILY_API_KEY），"
                    "再重新送出問題。"
                )
                yield new_messages[-1], "Web search failed"
                return

        if len(question.get("files", [])) > 0:
            file_path = question["files"][0]
            try:
                with open(file_path, "rb") as img_file:
                    encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
                    image_payload = {"images": [encoded_image]}
            except Exception as exc:  # noqa: BLE001
                logging.error("Image encode failed: %s", exc)
                image_payload = {}
        else:
            image_payload = {}

        # 2) First model call: decide if tool is needed
        tool_instruction = _build_tool_instruction()
        first_prompt = tool_instruction + "\n\n" + prompt
        first_message = {"role": "user", "content": first_prompt}
        if image_payload:
            first_message.update(image_payload)

        first_payload = {
            "model": model,
            "messages": [first_message],
            "options": options,
            "stream": False,
        }

        yield new_messages[-1], "Thinking: tool decision"
        try:
            first_response_text = _post_chat_once(url, headers, first_payload)
        except requests.RequestException as exc:
            logging.warning("Tool decision request failed, fallback to plain chat: %s", exc)
            yield new_messages[-1], "Tool decision failed, fallback to normal chat"

            fallback_payloads = _build_fallback_payloads(
                model=model,
                prompt=prompt,
                user_text=user_text,
                options=options,
                image_payload=image_payload,
            )

            fallback_error: requests.RequestException | None = None
            for attempt_name, fallback_payload in fallback_payloads:
                try:
                    if assistant_message["content"]:
                        assistant_message["content"] = ""
                        new_messages[-1]["content"] = ""

                    logging.info("Fallback chat attempt: %s", attempt_name)
                    for chunk in _stream_chat(url, headers, fallback_payload):
                        ensure_not_stopped()
                        assistant_message["content"] += chunk
                        new_messages[-1]["content"] = assistant_message["content"]
                        yield new_messages[-1], f"Generating answer ({attempt_name})"
                        time.sleep(0.02)

                    if assistant_message["content"].strip():
                        fallback_error = None
                        break
                except requests.RequestException as retry_exc:
                    fallback_error = retry_exc
                    logging.warning("Fallback chat attempt failed (%s): %s", attempt_name, retry_exc)

            if fallback_error is not None and not assistant_message["content"].strip():
                raise fallback_error

            if assistant_message["content"] and web_refs:
                linked = _linkify_reference_markers(assistant_message["content"], web_refs)
                linked_with_sources = _append_source_links(linked, web_refs)
                if linked_with_sources != assistant_message["content"]:
                    assistant_message["content"] = linked_with_sources
                    new_messages[-1]["content"] = linked_with_sources

            if not assistant_message["content"].strip():
                assistant_message["content"] = f"Server request failed: {exc}"
                new_messages[-1]["content"] = assistant_message["content"]
                yield new_messages[-1], f"Error: {exc}"
                return

            yield new_messages[-1], "Completed (fallback)"
            return

        extracted = tool_router.extract_tool_call(first_response_text)

        # 3) No tool call => return normal answer
        if extracted is None:
            clean_text = tool_router.remove_tool_call_markup(first_response_text)
            if not clean_text:
                clean_text = first_response_text.strip() or "(empty response)"

            assistant_message["content"] = clean_text
            if assistant_message["content"] and web_refs:
                linked = _linkify_reference_markers(assistant_message["content"], web_refs)
                assistant_message["content"] = _append_source_links(linked, web_refs)

            new_messages[-1]["content"] = assistant_message["content"]
            yield new_messages[-1], "Completed"
            return

        # 4) Tool execution
        yield new_messages[-1], f"Running tool: {extracted.get('name')}"
        tool_result = tool_router.execute_tool_call(extracted)

        # 5) Second model call: final answer from tool result
        second_prompt = (
            prompt
            + "\n\nTool call response:\n"
            + json.dumps(tool_result, ensure_ascii=False, indent=2)
            + "\n\nPlease explain the result to the user in natural language."
        )

        second_payload = {
            "model": model,
            "messages": [{"role": "user", "content": second_prompt}],
            "options": options,
            "stream": True,
        }

        yield new_messages[-1], "Generating final answer"
        for chunk in _stream_chat(url, headers, second_payload):
            ensure_not_stopped()
            assistant_message["content"] += chunk
            new_messages[-1]["content"] = assistant_message["content"]
            yield new_messages[-1], "Generating final answer"
            time.sleep(0.02)

        if assistant_message["content"] and web_refs:
            linked = _linkify_reference_markers(assistant_message["content"], web_refs)
            linked_with_sources = _append_source_links(linked, web_refs)
            if linked_with_sources != assistant_message["content"]:
                assistant_message["content"] = linked_with_sources
                new_messages[-1]["content"] = linked_with_sources

        yield new_messages[-1], "Completed"
    except requests.RequestException as exc:
        logging.error("Chat request failed: %s", exc)
        assistant_message["content"] = f"Server request failed: {exc}"
        yield new_messages[-1], f"Error: {exc}"
    except OperationCancelled:
        assistant_message["content"] = "已停止回答。"
        new_messages[-1]["content"] = assistant_message["content"]
        yield new_messages[-1], "Stopped"
    except Exception as exc:  # noqa: BLE001
        logging.error("Unexpected chat error: %s", exc)
        assistant_message["content"] = "Unexpected error."
        yield new_messages[-1], f"Error: {exc}"


def stop_response():
    request_stop()
    return gr.update(interactive=True), "Stopping..."


def send_prompt(prompt):
    return prompt


