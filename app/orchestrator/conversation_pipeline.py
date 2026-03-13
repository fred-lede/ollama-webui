from __future__ import annotations

import json
import time
from typing import Any, Callable, Generator

from app.core.cancellation import OperationCancelled


def run_legacy_conversation_stream(
    *,
    question: dict[str, Any],
    history: list[dict[str, str]],
    user_text: str,
    model: str,
    search_provider: str | None,
    web_search_enabled: bool,
    search_num_results: int | float | None,
    url: str,
    headers: dict[str, str],
    options: dict[str, float | int],
    assistant_message: dict[str, str],
    new_messages: list[dict[str, str]],
    ensure_not_stopped: Callable[[], None],
    build_web_context: Callable[[str, str | None, int | float | None], tuple[str | None, str | None, list[dict[str, str]]]],
    build_history_prompt: Callable[[list[dict[str, str]], str], str],
    build_tool_instruction: Callable[[], str],
    build_fallback_payloads: Callable[[str, str, str, dict[str, float | int], dict[str, object]], list[tuple[str, dict[str, object]]]],
    stream_chat: Callable[[str, dict[str, str], dict[str, Any]], Generator[str, None, None]],
    post_chat_once: Callable[[str, dict[str, str], dict[str, Any]], str],
    linkify_reference_markers: Callable[[str, list[dict[str, str]]], str],
    append_source_links: Callable[[str, list[dict[str, str]]], str],
    tool_router: Any,
    image_payload: dict[str, object],
) -> Generator[tuple[dict[str, str], str], None, None]:
    def run_fallback_generation(reason: str) -> Generator[tuple[dict[str, str], str], None, None]:
        yield new_messages[-1], reason

        fallback_payloads = build_fallback_payloads(
            model=model,
            prompt=prompt,
            user_text=user_text,
            options=options,
            image_payload=image_payload,
        )

        fallback_error: Exception | None = None
        for attempt_name, fallback_payload in fallback_payloads:
            try:
                if assistant_message["content"]:
                    assistant_message["content"] = ""
                    new_messages[-1]["content"] = ""

                for chunk in stream_chat(url, headers, fallback_payload):
                    ensure_not_stopped()
                    assistant_message["content"] += chunk
                    new_messages[-1]["content"] = assistant_message["content"]
                    yield new_messages[-1], f"Generating answer ({attempt_name})"
                    time.sleep(0.02)

                if assistant_message["content"].strip():
                    fallback_error = None
                    break
            except OperationCancelled:
                raise
            except Exception as retry_exc:  # noqa: BLE001
                fallback_error = retry_exc

        if fallback_error is not None and not assistant_message["content"].strip():
            raise fallback_error

        if assistant_message["content"] and web_refs:
            linked = linkify_reference_markers(assistant_message["content"], web_refs)
            linked_with_sources = append_source_links(linked, web_refs)
            if linked_with_sources != assistant_message["content"]:
                assistant_message["content"] = linked_with_sources
                new_messages[-1]["content"] = linked_with_sources

        if not assistant_message["content"].strip():
            assistant_message["content"] = "The model returned no content."
            new_messages[-1]["content"] = assistant_message["content"]
            yield new_messages[-1], "Empty model response"
            return

        yield new_messages[-1], "Completed (fallback)"

    prompt = build_history_prompt(history, user_text)
    web_refs: list[dict[str, str]] = []

    if bool(web_search_enabled) and user_text.strip():
        web_context, web_error, web_refs = build_web_context(user_text, search_provider, search_num_results)
        if web_context:
            prompt += (
                "\n\n" + web_context + "\n\n"
                "Instruction: Base your answer on the web search digest and cite source indices like [1], [2]."
            )
        elif web_error:
            assistant_message["content"] = (
                "網路搜尋已開啟，但搜尋失敗：" + str(web_error) + "。\n"
                "請確認已設定對應 API Key（Serper: SERPER_API_KEY / Tavily: TAVILY_API_KEY），"
                "再重新送出問題。"
            )
            yield new_messages[-1], "Web search failed"
            return

    tool_instruction = build_tool_instruction()
    first_prompt = tool_instruction + "\n\n" + prompt
    first_message: dict[str, object] = {"role": "user", "content": first_prompt}
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
        first_response_text = post_chat_once(url, headers, first_payload)
    except Exception as exc:
        try:
            yield from run_fallback_generation("Tool decision failed, fallback to normal chat")
        except OperationCancelled:
            raise
        except Exception:
            assistant_message["content"] = f"Server request failed: {exc}"
            new_messages[-1]["content"] = assistant_message["content"]
            yield new_messages[-1], f"Error: {exc}"
        return

    if not str(first_response_text or "").strip():
        yield from run_fallback_generation("Tool decision returned empty content, fallback to normal chat")
        return

    extracted = tool_router.extract_tool_call(first_response_text)

    if extracted is None:
        clean_text = tool_router.remove_tool_call_markup(first_response_text)
        if not clean_text:
            yield from run_fallback_generation("Tool decision produced no answer, fallback to normal chat")
            return

        assistant_message["content"] = clean_text
        if assistant_message["content"] and web_refs:
            linked = linkify_reference_markers(assistant_message["content"], web_refs)
            assistant_message["content"] = append_source_links(linked, web_refs)

        new_messages[-1]["content"] = assistant_message["content"]
        yield new_messages[-1], "Completed"
        return

    yield new_messages[-1], f"Running tool: {extracted.get('name')}"
    tool_result = tool_router.execute_tool_call(extracted)

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
    for chunk in stream_chat(url, headers, second_payload):
        ensure_not_stopped()
        assistant_message["content"] += chunk
        new_messages[-1]["content"] = assistant_message["content"]
        yield new_messages[-1], "Generating final answer"
        time.sleep(0.02)

    if assistant_message["content"] and web_refs:
        linked = linkify_reference_markers(assistant_message["content"], web_refs)
        linked_with_sources = append_source_links(linked, web_refs)
        if linked_with_sources != assistant_message["content"]:
            assistant_message["content"] = linked_with_sources
            new_messages[-1]["content"] = linked_with_sources

    yield new_messages[-1], "Completed"
