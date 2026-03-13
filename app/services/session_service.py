from __future__ import annotations

import copy
from typing import Any

from app.core.storage import JsonStore, generate_id, now_iso

SESSIONS_FILE = "sessions.json"


def _default_payload() -> dict[str, Any]:
    return {
        "current_session_id": None,
        "sessions": [],
    }


class SessionService:
    def __init__(self, store: JsonStore | None = None) -> None:
        self.store = store or JsonStore()
        self.store.ensure_file(SESSIONS_FILE, _default_payload())

    def _load(self) -> dict[str, Any]:
        payload = self.store.read(SESSIONS_FILE, _default_payload())
        sessions = payload.get("sessions", [])
        current_session_id = payload.get("current_session_id")
        if not isinstance(sessions, list):
            sessions = []
        return {
            "current_session_id": current_session_id,
            "sessions": sessions,
        }

    def _save(self, payload: dict[str, Any]) -> None:
        self.store.write(SESSIONS_FILE, payload)

    def list_sessions(self) -> list[dict[str, Any]]:
        payload = self._load()
        sessions = payload["sessions"]
        sessions.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
        return copy.deepcopy(sessions)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        payload = self._load()
        for session in payload["sessions"]:
            if session.get("id") == session_id:
                return copy.deepcopy(session)
        return None

    def get_current_session_id(self) -> str | None:
        payload = self._load()
        return payload["current_session_id"]

    def get_current_session(self) -> dict[str, Any] | None:
        session_id = self.get_current_session_id()
        if not session_id:
            return None
        return self.get_session(session_id)

    def create_session(
        self,
        title: str | None = None,
        *,
        persona_id: str | None = None,
        preset_id: str | None = None,
        server: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        payload = self._load()
        timestamp = now_iso()
        session = {
            "id": generate_id("sess"),
            "title": (title or "New Chat").strip() or "New Chat",
            "created_at": timestamp,
            "updated_at": timestamp,
            "persona_id": persona_id,
            "preset_id": preset_id,
            "server": server,
            "model": model,
            "messages": [],
        }
        payload["sessions"].append(session)
        payload["current_session_id"] = session["id"]
        self._save(payload)
        return copy.deepcopy(session)

    def rename_session(self, session_id: str, title: str) -> dict[str, Any] | None:
        normalized_title = title.strip()
        if not normalized_title:
            raise ValueError("Session title cannot be empty.")

        payload = self._load()
        for session in payload["sessions"]:
            if session.get("id") == session_id:
                session["title"] = normalized_title
                session["updated_at"] = now_iso()
                self._save(payload)
                return copy.deepcopy(session)
        return None

    def delete_session(self, session_id: str) -> bool:
        payload = self._load()
        sessions = payload["sessions"]
        original_count = len(sessions)
        sessions = [item for item in sessions if item.get("id") != session_id]
        if len(sessions) == original_count:
            return False

        payload["sessions"] = sessions
        if payload.get("current_session_id") == session_id:
            payload["current_session_id"] = sessions[0]["id"] if sessions else None
        self._save(payload)
        return True

    def set_current_session(self, session_id: str) -> dict[str, Any] | None:
        payload = self._load()
        for session in payload["sessions"]:
            if session.get("id") == session_id:
                payload["current_session_id"] = session_id
                self._save(payload)
                return copy.deepcopy(session)
        return None

    def append_message(self, session_id: str, role: str, content: str) -> dict[str, Any] | None:
        normalized_role = role.strip().lower()
        if normalized_role not in {"user", "assistant", "system", "tool"}:
            raise ValueError("Unsupported role.")

        payload = self._load()
        for session in payload["sessions"]:
            if session.get("id") == session_id:
                session.setdefault("messages", []).append(
                    {
                        "id": generate_id("msg"),
                        "role": normalized_role,
                        "content": content,
                        "created_at": now_iso(),
                    }
                )
                session["updated_at"] = now_iso()
                self._save(payload)
                return copy.deepcopy(session)
        return None

    def update_session(self, session_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        payload = self._load()
        for session in payload["sessions"]:
            if session.get("id") == session_id:
                for key, value in updates.items():
                    if key in {"id", "created_at", "messages"}:
                        continue
                    session[key] = value
                session["updated_at"] = now_iso()
                self._save(payload)
                return copy.deepcopy(session)
        return None
