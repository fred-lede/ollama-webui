from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.storage import JsonStore
from app.services import chat_service
from app.services.persona_service import PersonaService
from app.services.preset_service import PresetService
from app.services.prompt_service import PromptService
from app.services.session_service import SessionService


class ChatSessionPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = JsonStore(Path(self.tmpdir.name))
        self.session_service = SessionService(self.store)
        self.preset_service = PresetService(self.store)
        self.persona_service = PersonaService(self.store)
        self.prompt_service = PromptService(self.store)
        self.patches = [
            patch.object(chat_service, "session_service", self.session_service),
            patch.object(chat_service, "preset_service", self.preset_service),
            patch.object(chat_service, "persona_service", self.persona_service),
            patch.object(chat_service, "prompt_service", self.prompt_service),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self) -> None:
        for item in reversed(self.patches):
            item.stop()
        self.tmpdir.cleanup()

    def test_direct_tool_response_persists_full_exchange(self):
        with patch.object(chat_service.auto_tool_planner, "plan", return_value=chat_service.DirectAction(content="ok")):
            gen = chat_service.ask_question_stream(
                question={"text": "time?", "files": []},
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
            list(gen)

        current = self.session_service.get_current_session()
        self.assertIsNotNone(current)
        messages = current["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "time?")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "ok")

    def test_stopped_stream_does_not_persist_partial_assistant_message(self):
        def fake_stream(*_args, **_kwargs):
            yield "partial"
            chat_service.stop_response()
            yield "ignored"

        with patch.object(chat_service.auto_tool_planner, "plan", return_value=None), patch.object(
            chat_service, "_post_chat_once", side_effect=Exception("forced")
        ), patch.object(chat_service, "_stream_chat", side_effect=fake_stream):
            gen = chat_service.ask_question_stream(
                question={"text": "hello", "files": []},
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
            list(gen)

        current = self.session_service.get_current_session()
        self.assertIsNotNone(current)
        messages = current["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")

    def test_session_helpers_manage_choices_and_history(self):
        first = self.session_service.create_session("First Chat")
        self.session_service.append_message(first["id"], "user", "first")
        second = self.session_service.create_session("Second Chat")
        self.session_service.append_message(second["id"], "assistant", "second")

        choices, current_id = chat_service.list_chat_session_choices()
        self.assertEqual(current_id, second["id"])
        self.assertEqual(len(choices), 2)

        history, state, status = chat_service.switch_chat_session(first["id"])
        self.assertEqual(status, "Switched chat session.")
        self.assertEqual(history, state)
        self.assertEqual(history[0]["content"], "first")

        renamed_update, rename_status = chat_service.rename_chat_session(first["id"], "Renamed Chat")
        self.assertEqual(rename_status, "Chat renamed.")
        renamed_labels = [label for label, _value in renamed_update["choices"]]
        self.assertTrue(any(label.startswith("Renamed Chat") for label in renamed_labels))

        dropdown_update, history, state, delete_status = chat_service.delete_chat_session(first["id"])
        self.assertEqual(delete_status, "Chat deleted.")
        self.assertEqual(history, state)
        self.assertEqual(dropdown_update["value"], second["id"])

    def test_session_choice_label_is_compact(self):
        session = self.session_service.create_session("A very long chat title that should be truncated for the sidebar")
        label = chat_service.list_chat_session_choices()[0][0][0]
        self.assertIn(" | ", label)
        self.assertIn("...", label)
        self.assertNotIn(session["updated_at"], label)

    def test_create_new_chat_session_with_choices_returns_dropdown_update(self):
        dropdown_update, history, state, status = chat_service.create_new_chat_session_with_choices()
        self.assertEqual(status, "Started a new chat.")
        self.assertEqual(history, [])
        self.assertEqual(state, [])
        self.assertTrue(dropdown_update["choices"])
        self.assertIsNotNone(dropdown_update["value"])

    def test_switch_chat_session_with_state_syncs_preset_and_persona(self):
        preset = self.preset_service.create_preset(
            "Session Preset",
            {
                "llm_temperature": 0.3,
                "llm_max_tokens": 600,
                "llm_top_p": 0.85,
                "llm_typical_p": 0.6,
                "llm_num_ctx": 3000,
            },
        )
        persona = self.persona_service.create_persona(
            "Researcher",
            "You are a research assistant.",
            description="Focus on sources",
            default_model="research-model",
            default_preset=preset["id"],
        )
        session = self.session_service.create_session("Synced Chat")
        self.session_service.update_session(
            session["id"],
            {
                "preset_id": preset["id"],
                "persona_id": persona["id"],
                "model": "research-model",
            },
        )
        self.session_service.append_message(session["id"], "assistant", "saved")

        result = chat_service.switch_chat_session_with_state(session["id"])

        self.assertEqual(result[0][0]["content"], "saved")
        self.assertEqual(result[3], "Session Preset")
        self.assertEqual(result[5], "Researcher")
        self.assertEqual(result[6], "Focus on sources")
        self.assertEqual(result[7], "You are a research assistant.")
        self.assertEqual(result[8], "research-model")
        self.assertEqual(result[9], preset["id"])
        self.assertEqual(result[10], "research-model")
        self.assertEqual(result[11], 0.3)
        self.assertEqual(result[-1], "Switched chat session.")

    def test_prompt_library_helpers_cover_crud_and_insert(self):
        dropdown_update, status = chat_service.save_prompt_entry(
            None,
            "Translate",
            "translation",
            "Translate the following text to English.",
            True,
        )
        self.assertEqual(status, "Prompt created.")
        selected_id = dropdown_update["value"]
        self.assertIsNotNone(selected_id)

        loaded = chat_service.load_selected_prompt(selected_id)
        self.assertEqual(loaded[1], "Translate")
        self.assertEqual(loaded[2], "translation")
        self.assertTrue(loaded[4])

        dropdown_update, status = chat_service.save_prompt_entry(
            selected_id,
            "Translate Better",
            "translation",
            "Translate naturally.",
            False,
        )
        self.assertEqual(status, "Prompt updated.")

        workspace, status = chat_service.insert_selected_prompt_into_workspace(selected_id, "Existing text")
        self.assertIn("Existing text", workspace)
        self.assertIn("Translate naturally.", workspace)
        self.assertEqual(status, "Prompt inserted.")

        deleted = chat_service.delete_selected_prompt(selected_id)
        self.assertEqual(deleted[-1], "Prompt deleted.")

    def test_export_current_chat_markdown_writes_expected_content(self):
        session = self.session_service.create_session("Export Chat", model="qwen", server="http://127.0.0.1:11434")
        self.session_service.append_message(session["id"], "user", "hello")
        self.session_service.append_message(session["id"], "assistant", "world")

        export_dir = Path(self.tmpdir.name) / "exports"
        with patch.object(chat_service, "EXPORTS_DIR", export_dir):
            export_path, status = chat_service.export_current_chat_markdown()

        self.assertTrue(export_path.endswith(".md"))
        self.assertIn("Exported chat to", status)
        content = Path(export_path).read_text(encoding="utf-8")
        self.assertIn("# Export Chat", content)
        self.assertIn("## User", content)
        self.assertIn("hello", content)
        self.assertIn("## Assistant", content)
        self.assertIn("world", content)

    def test_apply_preset_updates_current_session_and_values(self):
        custom = self.preset_service.create_preset(
            "Low Temp",
            {
                "llm_temperature": 0.1,
                "llm_max_tokens": 512,
                "llm_top_p": 0.7,
                "llm_typical_p": 0.4,
                "llm_num_ctx": 2048,
            },
        )
        self.session_service.create_session("Preset Chat")

        result = chat_service.apply_preset_to_current_session(custom["id"])
        self.assertEqual(result[-1], "Applied preset: Low Temp")
        self.assertEqual(result[1], "Low Temp")
        self.assertEqual(result[2], 0.1)
        current = self.session_service.get_current_session()
        self.assertEqual(current["preset_id"], custom["id"])

    def test_persona_applies_default_model_and_prompt_in_chat(self):
        persona = self.persona_service.create_persona(
            "Coder",
            "You are a coding assistant.",
            default_model="coder-model",
            default_preset="balanced",
        )
        session = self.session_service.create_session("Persona Chat")
        self.session_service.update_session(session["id"], {"persona_id": persona["id"]})

        captured = {}

        def fake_stream(_url, _headers, payload):
            captured["payload"] = payload
            yield "answer"

        with patch.object(chat_service.auto_tool_planner, "plan", return_value=None), patch.object(
            chat_service, "_post_chat_once", side_effect=Exception("forced")
        ), patch.object(chat_service, "_stream_chat", side_effect=fake_stream):
            list(
                chat_service.ask_question_stream(
                    question={"text": "hello", "files": []},
                    history=[],
                    model="ui-model",
                    selected_server="http://127.0.0.1:11434",
                    llm_temperature=0.9,
                    llm_max_tokens=256,
                    llm_top_p=0.95,
                    llm_typical_p=0.8,
                    llm_num_ctx=1024,
                    search_provider="serper.dev",
                    web_search_enabled=False,
                    search_num_results=5,
                    search_summary_length="medium",
                )
            )

        payload = captured["payload"]
        self.assertEqual(payload["model"], "coder-model")
        self.assertIn("System: You are a coding assistant.", payload["messages"][0]["content"])
        self.assertEqual(payload["options"]["temperature"], 0.7)


if __name__ == "__main__":
    unittest.main()
