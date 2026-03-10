from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Generator

import requests

from app.core.cancellation import OperationCancelled, ensure_not_stopped, is_stop_requested


@dataclass(slots=True)
class ModelRuntime:
    timeout_seconds: int = 60

    def post_chat_once(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> str:
        ensure_not_stopped()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            body = response.json()
            return str(body.get("message", {}).get("content", ""))
        except requests.RequestException:
            # Some Ollama-compatible endpoints fail on non-stream mode.
            stream_payload = dict(payload)
            stream_payload["stream"] = True
            return "".join(self.stream_chat(url, headers, stream_payload))

    def stream_chat(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> Generator[str, None, None]:
        ensure_not_stopped()
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=self.timeout_seconds)
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


def build_fetch_summary_prompt(
    user_request: str,
    fetched_url: str,
    fetched_status: object,
    fetched_title: str,
    fetched_description: str,
    fetched_text: str,
) -> str:
    return (
        "You are summarizing a fetched webpage for the user.\n"
        "Return Traditional Chinese.\n"
        "Output format:\n"
        "1) 一句主旨\n"
        "2) 3-5點重點（條列）\n"
        "3) 若頁面可操作，補一行「可做的事」\n"
        "Keep concise and factual. Do not fabricate missing facts.\n\n"
        f"User request: {user_request}\n"
        f"URL: {fetched_url}\n"
        f"HTTP status: {fetched_status}\n"
        f"Page title: {fetched_title}\n"
        f"Meta description: {fetched_description}\n"
        "Fetched content:\n"
        f"{fetched_text[:3500]}\n"
    )


def build_search_summary_prompt(
    user_request: str,
    search_query: str,
    search_provider_name: str,
    compact_results_json: str,
    points_min: int,
    points_max: int,
    prompt_hint: str,
) -> str:
    return (
        "You are summarizing web search results for the user.\n"
        "Return Traditional Chinese.\n"
        "Output format:\n"
        "1) 一句結論\n"
        f"2) {points_min}-{points_max}點重點（條列）\n"
        "3) 資料來源（列出編號與網址）\n"
        f"{prompt_hint} Keep concise and factual. Do not fabricate missing facts.\n\n"
        f"User request: {user_request}\n"
        f"Search query: {search_query}\n"
        f"Search provider: {search_provider_name}\n"
        "Search results (JSON):\n"
        f"{compact_results_json}"
    )
