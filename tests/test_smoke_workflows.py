from __future__ import annotations

import unittest
from unittest.mock import patch

import requests

from app.core import cancellation
from app.orchestrator import (
    AutoToolPlanner,
    DirectAction,
    FetchForLlmAction,
    SearchForLlmAction,
    ToolIntentRouter,
)
from app.services import chat_service
from app.ui.gradio_app import clear_chat_history_state


class FakeRuntime:
    def __init__(self, response_map: dict[str, dict]):
        self.response_map = response_map

    def execute(self, name: str, arguments: dict):
        payload = self.response_map.get(name)
        if payload is None:
            return type("R", (), {"ok": False, "error": "missing fake response"})()
        return type(
            "R",
            (),
            {
                "ok": True,
                "tool_name": name,
                "result": payload,
                "error": None,
            },
        )()


class AutoToolPlannerSmokeTests(unittest.TestCase):
    def test_time_route_returns_direct_action(self):
        planner = AutoToolPlanner(
            ToolIntentRouter(),
            FakeRuntime({"datetime": {"formatted": "2026-03-10 23:00:00", "timezone": "Asia/Taipei", "iso": "x"}}),
        )
        action = planner.plan("現在幾點", "serper.dev", 5, "medium")
        self.assertIsInstance(action, DirectAction)
        self.assertIn("目前時間", action.content)

    def test_calculator_route_returns_direct_action(self):
        planner = AutoToolPlanner(
            ToolIntentRouter(),
            FakeRuntime({"calculator": {"expression": "2+3*4", "result": 14}}),
        )
        action = planner.plan("幫我算 2+3*4", "serper.dev", 5, "medium")
        self.assertIsInstance(action, DirectAction)
        self.assertIn("14", action.content)

    def test_fetch_route_returns_fetch_action(self):
        planner = AutoToolPlanner(
            ToolIntentRouter(),
            FakeRuntime(
                {
                    "fetch_url": {
                        "url": "https://example.com",
                        "status": 200,
                        "title": "Example",
                        "description": "Example description",
                        "text": "Example body text.",
                    }
                }
            ),
        )
        action = planner.plan("幫我摘要 https://example.com", "serper.dev", 5, "medium")
        self.assertIsInstance(action, FetchForLlmAction)
        self.assertEqual(action.url, "https://example.com")

    def test_search_route_returns_search_action(self):
        planner = AutoToolPlanner(
            ToolIntentRouter(),
            FakeRuntime(
                {
                    "web_search": {
                        "provider": "tavily",
                        "query": "ollama news",
                        "results": [
                            {
                                "title": "Blog · Ollama",
                                "snippet": "latest updates",
                                "url": "https://ollama.com/blog",
                            }
                        ],
                    }
                }
            ),
        )
        action = planner.plan("幫我搜尋 ollama news", "tavily", 5, "short")
        self.assertIsInstance(action, SearchForLlmAction)
        self.assertEqual(action.provider, "tavily")
        self.assertGreaterEqual(len(action.results), 1)


class StopAndClearSmokeTests(unittest.TestCase):
    def tearDown(self):
        cancellation.clear_stop()

    def test_stop_response_sets_stop_flag(self):
        cancellation.clear_stop()
        _, status = chat_service.stop_response()
        self.assertTrue(cancellation.is_stop_requested())
        self.assertEqual(status, "Stopping...")

    def test_ask_question_clears_stale_stop_flag_before_processing(self):
        cancellation.request_stop()
        with patch.object(chat_service.auto_tool_planner, "plan", return_value=DirectAction(content="ok")):
            gen = chat_service.ask_question_stream(
                question={"text": "現在幾點", "files": []},
                history=[],
                model="dummy",
                selected_server="http://127.0.0.1:11434",
                llm_temperature=0.5,
                llm_max_tokens=256,
                llm_top_p=0.9,
                llm_typical_p=0.7,
                llm_num_ctx=2048,
                search_provider="serper.dev",
                web_search_enabled=False,
                search_num_results=5,
                search_summary_length="medium",
            )
            first, status = next(gen)
            self.assertIn("ok", first.get("content", ""))
            self.assertEqual(status, "Tool intent executed")
            self.assertFalse(cancellation.is_stop_requested())

    def test_clear_history_helper(self):
        chat, state, status = clear_chat_history_state()
        self.assertEqual(chat, [])
        self.assertEqual(state, [])
        self.assertEqual(status, "")

    def test_stop_during_stream_returns_stopped_status(self):
        def fake_stream(*_args, **_kwargs):
            yield "first chunk"
            chat_service.stop_response()
            yield "second chunk"

        with patch.object(chat_service.auto_tool_planner, "plan", return_value=None), patch.object(
            chat_service, "_post_chat_once", side_effect=requests.RequestException("forced")
        ), patch.object(chat_service, "_stream_chat", side_effect=fake_stream):
            gen = chat_service.ask_question_stream(
                question={"text": "一般問題", "files": []},
                history=[],
                model="dummy",
                selected_server="http://127.0.0.1:11434",
                llm_temperature=0.5,
                llm_max_tokens=256,
                llm_top_p=0.9,
                llm_typical_p=0.7,
                llm_num_ctx=2048,
                search_provider="serper.dev",
                web_search_enabled=False,
                search_num_results=5,
                search_summary_length="medium",
            )
            statuses = []
            for _msg, st in gen:
                statuses.append(st)
            self.assertIn("Stopped", statuses)


if __name__ == "__main__":
    unittest.main()
