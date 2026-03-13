from __future__ import annotations

import copy
from typing import Any

from app.core.storage import JsonStore, generate_id

PERSONAS_FILE = "personas.json"


def _default_payload() -> dict[str, Any]:
    return {"personas": []}


class PersonaService:
    def __init__(self, store: JsonStore | None = None) -> None:
        self.store = store or JsonStore()
        self.store.ensure_file(PERSONAS_FILE, _default_payload())

    def _load(self) -> dict[str, Any]:
        payload = self.store.read(PERSONAS_FILE, _default_payload())
        personas = payload.get("personas", [])
        if not isinstance(personas, list):
            payload = _default_payload()
            self.store.write(PERSONAS_FILE, payload)
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        self.store.write(PERSONAS_FILE, payload)

    def list_personas(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._load()["personas"])

    def get_persona(self, persona_id: str) -> dict[str, Any] | None:
        for persona in self._load()["personas"]:
            if persona.get("id") == persona_id:
                return copy.deepcopy(persona)
        return None

    def create_persona(
        self,
        name: str,
        system_prompt: str,
        *,
        description: str = "",
        default_model: str | None = None,
        default_preset: str | None = None,
    ) -> dict[str, Any]:
        payload = self._load()
        persona = {
            "id": generate_id("persona"),
            "name": name.strip() or "New Persona",
            "description": description.strip(),
            "system_prompt": system_prompt,
            "default_model": default_model,
            "default_preset": default_preset,
        }
        payload["personas"].append(persona)
        self._save(payload)
        return copy.deepcopy(persona)

    def update_persona(self, persona_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        payload = self._load()
        for persona in payload["personas"]:
            if persona.get("id") == persona_id:
                persona.update({k: v for k, v in updates.items() if k != "id"})
                self._save(payload)
                return copy.deepcopy(persona)
        return None

    def delete_persona(self, persona_id: str) -> bool:
        payload = self._load()
        remaining = [persona for persona in payload["personas"] if persona.get("id") != persona_id]
        if len(remaining) == len(payload["personas"]):
            return False
        payload["personas"] = remaining
        self._save(payload)
        return True

    def apply_persona(self, persona_id: str, context: dict[str, Any] | None = None) -> dict[str, Any] | None:
        persona = self.get_persona(persona_id)
        if persona is None:
            return None
        merged = dict(context or {})
        merged["system_prompt"] = persona.get("system_prompt", "")
        if persona.get("default_model"):
            merged["model"] = persona["default_model"]
        if persona.get("default_preset"):
            merged["preset_id"] = persona["default_preset"]
        return merged
