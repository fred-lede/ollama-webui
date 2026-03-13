from __future__ import annotations

import copy
from typing import Any

from app.core.storage import JsonStore, generate_id

PRESETS_FILE = "presets.json"

BUILTIN_PRESETS: list[dict[str, Any]] = [
    {
        "id": "precise",
        "name": "Precise",
        "llm_temperature": 0.2,
        "llm_max_tokens": 1024,
        "llm_top_p": 0.8,
        "llm_typical_p": 0.5,
        "llm_num_ctx": 4096,
        "builtin": True,
    },
    {
        "id": "balanced",
        "name": "Balanced",
        "llm_temperature": 0.7,
        "llm_max_tokens": 2048,
        "llm_top_p": 0.9,
        "llm_typical_p": 0.7,
        "llm_num_ctx": 4096,
        "builtin": True,
    },
    {
        "id": "creative",
        "name": "Creative",
        "llm_temperature": 0.95,
        "llm_max_tokens": 3072,
        "llm_top_p": 1.0,
        "llm_typical_p": 0.95,
        "llm_num_ctx": 4096,
        "builtin": True,
    },
    {
        "id": "longform",
        "name": "Longform",
        "llm_temperature": 0.6,
        "llm_max_tokens": 4096,
        "llm_top_p": 0.9,
        "llm_typical_p": 0.7,
        "llm_num_ctx": 8192,
        "builtin": True,
    },
]


def _default_payload() -> dict[str, Any]:
    return {"presets": copy.deepcopy(BUILTIN_PRESETS)}


class PresetService:
    def __init__(self, store: JsonStore | None = None) -> None:
        self.store = store or JsonStore()
        self.store.ensure_file(PRESETS_FILE, _default_payload())

    def _load(self) -> dict[str, Any]:
        payload = self.store.read(PRESETS_FILE, _default_payload())
        presets = payload.get("presets", [])
        if not isinstance(presets, list) or not presets:
            payload = _default_payload()
            self.store.write(PRESETS_FILE, payload)
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        self.store.write(PRESETS_FILE, payload)

    def list_presets(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._load()["presets"])

    def get_preset(self, preset_id: str) -> dict[str, Any] | None:
        for preset in self._load()["presets"]:
            if preset.get("id") == preset_id:
                return copy.deepcopy(preset)
        return None

    def create_preset(self, name: str, parameters: dict[str, Any]) -> dict[str, Any]:
        payload = self._load()
        preset = {
            "id": generate_id("preset"),
            "name": name.strip() or "Custom Preset",
            "llm_temperature": parameters.get("llm_temperature", 0.7),
            "llm_max_tokens": parameters.get("llm_max_tokens", 2048),
            "llm_top_p": parameters.get("llm_top_p", 0.9),
            "llm_typical_p": parameters.get("llm_typical_p", 0.7),
            "llm_num_ctx": parameters.get("llm_num_ctx", 4096),
            "builtin": False,
        }
        payload["presets"].append(preset)
        self._save(payload)
        return copy.deepcopy(preset)

    def update_preset(self, preset_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        payload = self._load()
        for preset in payload["presets"]:
            if preset.get("id") == preset_id:
                if preset.get("builtin"):
                    raise ValueError("Builtin presets cannot be modified.")
                preset.update({k: v for k, v in updates.items() if k != "id"})
                self._save(payload)
                return copy.deepcopy(preset)
        return None

    def delete_preset(self, preset_id: str) -> bool:
        payload = self._load()
        for preset in payload["presets"]:
            if preset.get("id") == preset_id and preset.get("builtin"):
                return False
        remaining = [preset for preset in payload["presets"] if preset.get("id") != preset_id]
        if len(remaining) == len(payload["presets"]):
            return False
        payload["presets"] = remaining
        self._save(payload)
        return True

    def apply_preset(self, preset_id: str, base_parameters: dict[str, Any] | None = None) -> dict[str, Any] | None:
        preset = self.get_preset(preset_id)
        if preset is None:
            return None
        merged = dict(base_parameters or {})
        for key in ("llm_temperature", "llm_max_tokens", "llm_top_p", "llm_typical_p", "llm_num_ctx"):
            merged[key] = preset[key]
        return merged
