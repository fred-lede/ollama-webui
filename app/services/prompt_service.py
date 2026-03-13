from __future__ import annotations

import copy
from typing import Any

from app.core.storage import JsonStore, generate_id, now_iso

PROMPTS_FILE = "prompts.json"


def _default_payload() -> dict[str, Any]:
    return {"prompts": []}


class PromptService:
    def __init__(self, store: JsonStore | None = None) -> None:
        self.store = store or JsonStore()
        self.store.ensure_file(PROMPTS_FILE, _default_payload())

    def _load(self) -> dict[str, Any]:
        payload = self.store.read(PROMPTS_FILE, _default_payload())
        prompts = payload.get("prompts", [])
        if not isinstance(prompts, list):
            payload = _default_payload()
            self.store.write(PROMPTS_FILE, payload)
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        self.store.write(PROMPTS_FILE, payload)

    def list_prompts(self) -> list[dict[str, Any]]:
        prompts = self._load()["prompts"]
        prompts.sort(key=lambda item: (not bool(item.get("favorite")), str(item.get("name", "")).lower()))
        return copy.deepcopy(prompts)

    def get_prompt(self, prompt_id: str) -> dict[str, Any] | None:
        for prompt in self._load()["prompts"]:
            if prompt.get("id") == prompt_id:
                return copy.deepcopy(prompt)
        return None

    def create_prompt(
        self,
        name: str,
        content: str,
        *,
        category: str = "general",
        favorite: bool = False,
    ) -> dict[str, Any]:
        payload = self._load()
        timestamp = now_iso()
        prompt = {
            "id": generate_id("prompt"),
            "name": name.strip() or "Untitled Prompt",
            "category": category.strip() or "general",
            "content": content,
            "favorite": bool(favorite),
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        payload["prompts"].append(prompt)
        self._save(payload)
        return copy.deepcopy(prompt)

    def update_prompt(self, prompt_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        payload = self._load()
        for prompt in payload["prompts"]:
            if prompt.get("id") == prompt_id:
                prompt.update({k: v for k, v in updates.items() if k != "id"})
                prompt["updated_at"] = now_iso()
                self._save(payload)
                return copy.deepcopy(prompt)
        return None

    def delete_prompt(self, prompt_id: str) -> bool:
        payload = self._load()
        remaining = [prompt for prompt in payload["prompts"] if prompt.get("id") != prompt_id]
        if len(remaining) == len(payload["prompts"]):
            return False
        payload["prompts"] = remaining
        self._save(payload)
        return True
