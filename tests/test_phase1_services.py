from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.core.storage import JsonStore
from app.services.persona_service import PersonaService
from app.services.preset_service import PresetService
from app.services.prompt_service import PromptService
from app.services.session_service import SessionService


class Phase1ServicesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = JsonStore(Path(self.tmpdir.name))

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_store_initializes_expected_files(self):
        SessionService(self.store)
        PromptService(self.store)
        PersonaService(self.store)
        PresetService(self.store)

        self.assertTrue((Path(self.tmpdir.name) / "sessions.json").exists())
        self.assertTrue((Path(self.tmpdir.name) / "prompts.json").exists())
        self.assertTrue((Path(self.tmpdir.name) / "personas.json").exists())
        self.assertTrue((Path(self.tmpdir.name) / "presets.json").exists())

    def test_session_crud_and_current_session(self):
        service = SessionService(self.store)

        first = service.create_session("First Chat", persona_id="coder")
        second = service.create_session("Second Chat")

        self.assertEqual(service.get_current_session_id(), second["id"])
        self.assertEqual(len(service.list_sessions()), 2)

        renamed = service.rename_session(first["id"], "Renamed Chat")
        self.assertIsNotNone(renamed)
        self.assertEqual(renamed["title"], "Renamed Chat")

        updated = service.append_message(first["id"], "user", "hello")
        self.assertIsNotNone(updated)
        self.assertEqual(len(updated["messages"]), 1)

        current = service.set_current_session(first["id"])
        self.assertIsNotNone(current)
        self.assertEqual(service.get_current_session_id(), first["id"])

        deleted = service.delete_session(first["id"])
        self.assertTrue(deleted)
        self.assertEqual(len(service.list_sessions()), 1)
        self.assertEqual(service.get_current_session_id(), second["id"])

    def test_preset_service_loads_builtins_and_custom_crud(self):
        service = PresetService(self.store)

        presets = service.list_presets()
        self.assertGreaterEqual(len(presets), 4)
        self.assertIsNotNone(service.get_preset("balanced"))

        custom = service.create_preset(
            "Low Temp",
            {
                "llm_temperature": 0.1,
                "llm_max_tokens": 512,
                "llm_top_p": 0.7,
                "llm_typical_p": 0.4,
                "llm_num_ctx": 2048,
            },
        )
        applied = service.apply_preset(custom["id"], {"foo": "bar"})
        self.assertIsNotNone(applied)
        self.assertEqual(applied["foo"], "bar")
        self.assertEqual(applied["llm_temperature"], 0.1)

        updated = service.update_preset(custom["id"], {"name": "Very Low Temp"})
        self.assertIsNotNone(updated)
        self.assertEqual(updated["name"], "Very Low Temp")
        self.assertTrue(service.delete_preset(custom["id"]))
        self.assertFalse(service.delete_preset("balanced"))

    def test_persona_service_crud_and_apply(self):
        service = PersonaService(self.store)

        persona = service.create_persona(
            "Coder",
            "You are a coding assistant.",
            description="Writes code",
            default_model="qwen",
            default_preset="balanced",
        )
        self.assertEqual(len(service.list_personas()), 1)

        applied = service.apply_persona(persona["id"], {"foo": "bar"})
        self.assertIsNotNone(applied)
        self.assertEqual(applied["system_prompt"], "You are a coding assistant.")
        self.assertEqual(applied["model"], "qwen")
        self.assertEqual(applied["preset_id"], "balanced")

        updated = service.update_persona(persona["id"], {"name": "Senior Coder"})
        self.assertIsNotNone(updated)
        self.assertEqual(updated["name"], "Senior Coder")
        self.assertTrue(service.delete_persona(persona["id"]))

    def test_prompt_service_crud(self):
        service = PromptService(self.store)

        prompt = service.create_prompt(
            "Translate",
            "Translate the following text.",
            category="translation",
            favorite=True,
        )
        prompts = service.list_prompts()
        self.assertEqual(len(prompts), 1)
        self.assertTrue(prompts[0]["favorite"])

        updated = service.update_prompt(prompt["id"], {"name": "Translate to English", "favorite": False})
        self.assertIsNotNone(updated)
        self.assertEqual(updated["name"], "Translate to English")
        self.assertFalse(updated["favorite"])
        self.assertTrue(service.delete_prompt(prompt["id"]))


if __name__ == "__main__":
    unittest.main()
