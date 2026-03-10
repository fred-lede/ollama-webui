from __future__ import annotations

import base64
import json
import logging
import re
import threading
import time
from urllib.parse import urlparse

import gradio as gr
import requests

from app.core.tool_router import ToolRouter
from app.tools import build_default_registry

stop_event = threading.Event()
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
    keywords = _extract_search_keywords(question_text)
    try:
        num_results = int(search_num_results or 5)
    except (TypeError, ValueError):
        num_results = 5
    num_results = max(1, min(num_results, 10))

    result = tool_registry.execute(
        "web_search",
        {
            "query": keywords,
            "num_results": num_results,
            "provider": search_provider or "serper.dev",
        },
    )

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


def _build_history_prompt(history: list[dict[str, str]], user_text: str) -> str:
    prompt = "Previous conversation:\n" if history else ""
    for msg in history:
        prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    prompt += f"\nUser: {user_text}\n"
    return prompt


def _build_ollama_options(llm_temperature, llm_max_tokens) -> dict[str, float | int]:
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
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    body = response.json()
    return str(body.get("message", {}).get("content", ""))


def _stream_chat(url: str, headers: dict[str, str], payload: dict):
    response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
    response.raise_for_status()
    for line in response.iter_lines():
        if stop_event.is_set():
            break
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
            content = data.get("message", {}).get("content", "")
            if content:
                yield content
        except (json.JSONDecodeError, KeyError, TypeError):
            continue


def ask_question_stream(
    question,
    history,
    model,
    selected_server,
    llm_temperature,
    llm_max_tokens,
    search_provider,
    web_search_enabled,
    search_num_results,
):
    stop_event.clear()

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

    if model is None:
        assistant_message["content"] = "Please select a model."
        yield new_messages[-1], "Missing model"
        return

    if selected_server is None:
        assistant_message["content"] = "Please select a server."
        yield new_messages[-1], "Missing server"
        return

    try:
        url = _build_chat_url(str(selected_server))
        headers = {"Content-Type": "application/json"}
        options = _build_ollama_options(llm_temperature, llm_max_tokens)

        prompt = _build_history_prompt(history, user_text)

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

            fallback_message = {"role": "user", "content": prompt}
            if image_payload:
                fallback_message.update(image_payload)

            fallback_payload = {
                "model": model,
                "messages": [fallback_message],
                "options": options,
                "stream": True,
            }

            for chunk in _stream_chat(url, headers, fallback_payload):
                assistant_message["content"] += chunk
                new_messages[-1]["content"] = assistant_message["content"]
                yield new_messages[-1], "Generating answer (fallback)"
                time.sleep(0.02)

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
    except Exception as exc:  # noqa: BLE001
        logging.error("Unexpected chat error: %s", exc)
        assistant_message["content"] = "Unexpected error."
        yield new_messages[-1], f"Error: {exc}"


def stop_response():
    stop_event.set()
    return gr.update(interactive=True)


def send_prompt(prompt):
    return prompt


