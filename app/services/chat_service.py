from __future__ import annotations

import base64
import json
import logging
import re
import time
from urllib.parse import urlparse

import gradio as gr
import requests

from app.core.cancellation import (
    OperationCancelled,
    clear_stop,
    ensure_not_stopped,
    request_stop,
)
from app.orchestrator import (
    AutoToolAction,
    AutoToolPlanner,
    DirectAction,
    FetchForLlmAction,
    SearchForLlmAction,
    ToolIntentRouter,
    ToolPolicy,
    ToolRuntime,
    clean_search_snippet,
    normalize_summary_length,
    summary_style_config,
)
from app.orchestrator.model_runtime import (
    ModelRuntime,
    build_fetch_summary_prompt,
    build_search_summary_prompt,
)
from app.orchestrator.conversation_pipeline import run_legacy_conversation_stream
from app.core.tool_router import ToolRouter
from app.tools import build_default_registry

tool_registry = build_default_registry()
tool_router = ToolRouter(tool_registry)
tool_policy = ToolPolicy()
tool_intent_router = ToolIntentRouter()
tool_runtime = ToolRuntime(tool_registry, tool_policy)
auto_tool_planner = AutoToolPlanner(tool_intent_router, tool_runtime)
model_runtime = ModelRuntime(timeout_seconds=60)

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


def _try_auto_tool_intent(
    text: str | None,
    search_provider: str | None,
    search_num_results: int | float | None,
    search_summary_length: str | None,
) -> AutoToolAction | None:
    return auto_tool_planner.plan(
        text=text,
        search_provider=search_provider,
        search_num_results=search_num_results,
        search_summary_length=search_summary_length,
    )


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
    return model_runtime.post_chat_once(url, headers, payload)


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
    yield from model_runtime.stream_chat(url, headers, payload)


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
    if isinstance(auto_tool_action, DirectAction):
        assistant_message["content"] = auto_tool_action.content.strip() or "Tool executed."
        yield new_messages[-1], "Tool intent executed"
        return

    if model is None:
        if isinstance(auto_tool_action, FetchForLlmAction):
            lines = [f"已抓取網址：{auto_tool_action.url}", f"狀態碼：{auto_tool_action.status}"]
            if auto_tool_action.title:
                lines.append(f"標題：{auto_tool_action.title}")
            lines.append(f"內容摘要：{auto_tool_action.fallback_summary}")
            assistant_message["content"] = "\n".join(lines)
            yield new_messages[-1], "Tool intent executed (fallback summary)"
            return
        if isinstance(auto_tool_action, SearchForLlmAction):
            assistant_message["content"] = auto_tool_action.fallback_content.strip() or "搜尋完成。"
            yield new_messages[-1], "Tool intent executed (fallback summary)"
            return
        assistant_message["content"] = "Please select a model."
        yield new_messages[-1], "Missing model"
        return

    if selected_server is None:
        if isinstance(auto_tool_action, FetchForLlmAction):
            lines = [f"已抓取網址：{auto_tool_action.url}", f"狀態碼：{auto_tool_action.status}"]
            if auto_tool_action.title:
                lines.append(f"標題：{auto_tool_action.title}")
            lines.append(f"內容摘要：{auto_tool_action.fallback_summary}")
            assistant_message["content"] = "\n".join(lines)
            yield new_messages[-1], "Tool intent executed (fallback summary)"
            return
        if isinstance(auto_tool_action, SearchForLlmAction):
            assistant_message["content"] = auto_tool_action.fallback_content.strip() or "搜尋完成。"
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

        if isinstance(auto_tool_action, FetchForLlmAction):
            fetched_url = auto_tool_action.url
            fetched_status = auto_tool_action.status
            fetched_title = auto_tool_action.title
            fetched_description = auto_tool_action.description
            fetched_text = auto_tool_action.text
            fallback_summary = auto_tool_action.fallback_summary

            summary_prompt = build_fetch_summary_prompt(
                user_request=user_text,
                fetched_url=fetched_url,
                fetched_status=fetched_status,
                fetched_title=fetched_title,
                fetched_description=fetched_description,
                fetched_text=fetched_text,
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

        if isinstance(auto_tool_action, SearchForLlmAction):
            search_provider_name = auto_tool_action.provider or str(search_provider or "serper.dev").strip()
            search_query = auto_tool_action.query or user_text
            summary_length = normalize_summary_length(auto_tool_action.summary_length or str(search_summary_length))
            style_cfg = summary_style_config(summary_length)
            points_min = int(style_cfg["points_min"])
            points_max = int(style_cfg["points_max"])
            prompt_hint = str(style_cfg["prompt_hint"])
            search_results = auto_tool_action.results
            fallback_content = auto_tool_action.fallback_content.strip()

            if isinstance(search_results, list):
                compact_results = [
                    {
                        "title": str(item.get("title", "")).strip(),
                        "snippet": clean_search_snippet(str(item.get("snippet", "")), max_len=260),
                        "url": str(item.get("url", "")).strip(),
                    }
                    for item in search_results
                ]
            else:
                compact_results = []

            summary_prompt = build_search_summary_prompt(
                user_request=user_text,
                search_query=search_query,
                search_provider_name=search_provider_name,
                compact_results_json=json.dumps(compact_results, ensure_ascii=False, indent=2),
                points_min=points_min,
                points_max=points_max,
                prompt_hint=prompt_hint,
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
        yield from run_legacy_conversation_stream(
            question=question,
            history=history,
            user_text=user_text,
            model=model,
            search_provider=search_provider,
            web_search_enabled=bool(web_search_enabled),
            search_num_results=search_num_results,
            url=url,
            headers=headers,
            options=options,
            assistant_message=assistant_message,
            new_messages=new_messages,
            ensure_not_stopped=ensure_not_stopped,
            build_web_context=_build_web_context,
            build_history_prompt=_build_history_prompt,
            build_tool_instruction=_build_tool_instruction,
            build_fallback_payloads=_build_fallback_payloads,
            stream_chat=_stream_chat,
            post_chat_once=_post_chat_once,
            linkify_reference_markers=_linkify_reference_markers,
            append_source_links=_append_source_links,
            tool_router=tool_router,
            image_payload=image_payload,
        )
        return
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


