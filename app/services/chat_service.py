from __future__ import annotations

import base64
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import gradio as gr
import requests

from app.core.cancellation import (
    OperationCancelled,
    clear_stop,
    ensure_not_stopped,
    request_stop,
)
from app.orchestrator import (
    AutoToolAction,
    AutoToolPlanner,
    DirectAction,
    FetchForLlmAction,
    SearchForLlmAction,
    ToolIntentRouter,
    ToolPolicy,
    ToolRuntime,
    clean_search_snippet,
    normalize_summary_length,
    summary_style_config,
)
from app.orchestrator.model_runtime import (
    ModelRuntime,
    build_fetch_summary_prompt,
    build_search_summary_prompt,
)
from app.orchestrator.conversation_pipeline import run_legacy_conversation_stream
from app.core.tool_router import ToolRouter
from app.services.persona_service import PersonaService
from app.services.preset_service import PresetService
from app.services.prompt_service import PromptService
from app.services.session_service import SessionService
from app.tools import build_default_registry

tool_registry = build_default_registry()
tool_router = ToolRouter(tool_registry)
tool_policy = ToolPolicy()
tool_intent_router = ToolIntentRouter()
tool_runtime = ToolRuntime(tool_registry, tool_policy)
auto_tool_planner = AutoToolPlanner(tool_intent_router, tool_runtime)
model_runtime = ModelRuntime(timeout_seconds=60)
session_service = SessionService()
preset_service = PresetService()
persona_service = PersonaService()
prompt_service = PromptService()
EXPORTS_DIR = Path(__file__).resolve().parents[2] / "exports"
_SESSION_LABEL_LANGUAGE = "English"

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


def _get_or_create_current_session() -> dict[str, object]:
    session = session_service.get_current_session()
    if session is not None:
        return session
    return session_service.create_session()


def _session_messages_to_history(session: dict[str, object] | None) -> list[dict[str, str]]:
    if not session:
        return []

    raw_messages = session.get("messages", [])
    if not isinstance(raw_messages, list):
        return []

    history: list[dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        if role not in {"user", "assistant", "system", "tool"}:
            continue
        history.append({"role": role, "content": str(item.get("content", ""))})
    return history


def load_current_chat_history() -> list[dict[str, str]]:
    return _session_messages_to_history(_get_or_create_current_session())


def create_new_chat_session() -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    session_service.create_session()
    return [], [], "Started a new chat."


def set_session_label_language(language: str | None) -> None:
    global _SESSION_LABEL_LANGUAGE
    normalized = str(language or "").strip()
    _SESSION_LABEL_LANGUAGE = normalized or "English"


def _localized_new_chat_title(title: str) -> str:
    normalized = title.strip() or "New Chat"
    if normalized != "New Chat":
        return normalized

    if _SESSION_LABEL_LANGUAGE == "Chinese":
        return "新增對話"
    if _SESSION_LABEL_LANGUAGE == "Thailand":
        return "แชตใหม่"
    return "New Chat"


def _build_session_choice_label(session: dict[str, object]) -> str:
    title = _localized_new_chat_title(str(session.get("title", "New Chat")))
    if len(title) > 30:
        title = title[:27].rstrip() + "..."
    updated_at = _format_session_time(str(session.get("updated_at", "")).strip())
    base = f"{title} | {updated_at}" if updated_at else title
    if session.get("pinned"):
        return f"[PIN] {base}"
    return base


def _format_session_time(value: str) -> str:
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(value)
        if _SESSION_LABEL_LANGUAGE == "Chinese":
            return f"{dt.month}月{dt.day}日 {dt.strftime('%H:%M')}"
        if _SESSION_LABEL_LANGUAGE == "Thailand":
            thai_months = [
                "",
                "ม.ค.",
                "ก.พ.",
                "มี.ค.",
                "เม.ย.",
                "พ.ค.",
                "มิ.ย.",
                "ก.ค.",
                "ส.ค.",
                "ก.ย.",
                "ต.ค.",
                "พ.ย.",
                "ธ.ค.",
            ]
            return f"{dt.day} {thai_months[dt.month]} {dt.strftime('%H:%M')}"
        return dt.strftime("%b %d %H:%M")
    except ValueError:
        return value[:16]


def list_chat_session_choices() -> tuple[list[tuple[str, str]], str | None]:
    sessions = session_service.list_sessions()
    choices = [(_build_session_choice_label(session), str(session["id"])) for session in sessions]
    return choices, session_service.get_current_session_id()


def _session_dataset_update() -> tuple[dict, str | None]:
    choices, current_id = list_chat_session_choices()
    labels = [label for label, _value in choices]
    samples = [[value] for _label, value in choices]
    return gr.update(samples=samples, sample_labels=labels), current_id


def _session_id_from_dataset_index(index: int | tuple[int, ...] | None) -> str | None:
    choices, _current_id = list_chat_session_choices()
    if isinstance(index, tuple):
        if not index:
            return None
        index = index[0]
    if not isinstance(index, int):
        return None
    if index < 0 or index >= len(choices):
        return None
    return choices[index][1]


def switch_chat_session(session_id: str | None) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    if not session_id:
        history = load_current_chat_history()
        return history, history, "No session selected."

    session = session_service.set_current_session(session_id)
    if session is None:
        history = load_current_chat_history()
        return history, history, "Session not found."

    history = _session_messages_to_history(session)
    return history, history, "Switched chat session."


def switch_chat_session_with_state(
    session_id: str | None,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict,
    str,
    dict,
    str,
    str,
    str,
    str | None,
    str | None,
    str | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    str,
]:
    status = "Switched chat session."
    if not session_id:
        session = _get_or_create_current_session()
        status = "No session selected."
    else:
        session = session_service.set_current_session(session_id)
        if session is None:
            session = _get_or_create_current_session()
            status = "Session not found."

    state = _resolve_session_ui_state(session)
    history = state["history"]
    return (
        history,
        history,
        state["preset_dropdown"],
        state["preset_name"],
        state["persona_dropdown"],
        state["persona_name"],
        state["persona_description"],
        state["persona_system_prompt"],
        state["persona_default_model"],
        state["persona_default_preset"],
        state["model"],
        state["llm_temperature"],
        state["llm_max_tokens"],
        state["llm_top_p"],
        state["llm_typical_p"],
        state["llm_num_ctx"],
        state["llm_temperature"],
        state["llm_max_tokens"],
        state["llm_top_p"],
        state["llm_typical_p"],
        state["llm_num_ctx"],
        status,
    )


def switch_chat_session_from_dataset(
    evt: gr.SelectData,
) -> tuple[
    str | None,
    list[dict[str, str]],
    list[dict[str, str]],
    dict,
    str,
    dict,
    str,
    str,
    str,
    str | None,
    str | None,
    str | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    str,
]:
    session_id = _session_id_from_dataset_index(getattr(evt, "index", None))
    result = switch_chat_session_with_state(session_id)
    return (session_id, *result)


def create_new_chat_session_with_choices() -> tuple[dict, list[dict[str, str]], list[dict[str, str]], str]:
    create_new_chat_session()
    choices, current_id = list_chat_session_choices()
    return gr.update(choices=choices, value=current_id), [], [], "Started a new chat."


def create_new_chat_session_with_state() -> tuple[
    dict,
    list[dict[str, str]],
    list[dict[str, str]],
    dict,
    str,
    dict,
    str,
    str,
    str,
    str | None,
    str | None,
    str | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    str,
]:
    session = session_service.create_session()
    session_choices, current_session_id = list_chat_session_choices()
    state = _resolve_session_ui_state(session)
    history = state["history"]
    return (
        gr.update(choices=session_choices, value=current_session_id),
        history,
        history,
        state["preset_dropdown"],
        state["preset_name"],
        state["persona_dropdown"],
        state["persona_name"],
        state["persona_description"],
        state["persona_system_prompt"],
        state["persona_default_model"],
        state["persona_default_preset"],
        state["model"],
        state["llm_temperature"],
        state["llm_max_tokens"],
        state["llm_top_p"],
        state["llm_typical_p"],
        state["llm_num_ctx"],
        state["llm_temperature"],
        state["llm_max_tokens"],
        state["llm_top_p"],
        state["llm_typical_p"],
        state["llm_num_ctx"],
        "Started a new chat.",
    )


def create_new_chat_session_with_dataset_state() -> tuple[
    dict,
    str | None,
    list[dict[str, str]],
    list[dict[str, str]],
    dict,
    str,
    dict,
    str,
    str,
    str,
    str | None,
    str | None,
    str | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    str,
]:
    session = session_service.create_session()
    dataset_update, current_session_id = _session_dataset_update()
    state = _resolve_session_ui_state(session)
    history = state["history"]
    return (
        dataset_update,
        current_session_id,
        history,
        history,
        state["preset_dropdown"],
        state["preset_name"],
        state["persona_dropdown"],
        state["persona_name"],
        state["persona_description"],
        state["persona_system_prompt"],
        state["persona_default_model"],
        state["persona_default_preset"],
        state["model"],
        state["llm_temperature"],
        state["llm_max_tokens"],
        state["llm_top_p"],
        state["llm_typical_p"],
        state["llm_num_ctx"],
        state["llm_temperature"],
        state["llm_max_tokens"],
        state["llm_top_p"],
        state["llm_typical_p"],
        state["llm_num_ctx"],
        "Started a new chat.",
    )


def clear_current_chat_with_state() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict,
    str,
    dict,
    str,
    str,
    str,
    str | None,
    str | None,
    str | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    str,
]:
    session = _get_or_create_current_session()
    cleared = session_service.clear_messages(str(session["id"]))
    active_session = cleared or session
    state = _resolve_session_ui_state(active_session)
    history = state["history"]
    status = "Current chat cleared." if cleared is not None else "Session not found."
    return (
        history,
        history,
        state["preset_dropdown"],
        state["preset_name"],
        state["persona_dropdown"],
        state["persona_name"],
        state["persona_description"],
        state["persona_system_prompt"],
        state["persona_default_model"],
        state["persona_default_preset"],
        state["model"],
        state["llm_temperature"],
        state["llm_max_tokens"],
        state["llm_top_p"],
        state["llm_typical_p"],
        state["llm_num_ctx"],
        state["llm_temperature"],
        state["llm_max_tokens"],
        state["llm_top_p"],
        state["llm_typical_p"],
        state["llm_num_ctx"],
        status,
    )


def set_session_pinned(session_id: str | None, pinned: bool) -> str:
    if not session_id:
        return "No session selected."
    updated = session_service.set_pinned(session_id, pinned)
    if updated is None:
        return "Session not found."
    return "Session pinned." if pinned else "Session unpinned."


def is_session_pinned(session_id: str | None) -> bool:
    if not session_id:
        return False
    session = session_service.get_session(session_id)
    return bool(session.get("pinned")) if session else False


def get_current_session_preferences() -> dict[str, str | None]:
    session = _get_or_create_current_session()
    return {
        "server": str(session.get("server") or "").strip() or None,
        "model": str(session.get("model") or "").strip() or None,
        "preset_id": str(session.get("preset_id") or "").strip() or None,
        "persona_id": str(session.get("persona_id") or "").strip() or None,
    }


def update_current_session_preferences(
    *,
    server: str | None = None,
    model: str | None = None,
    preset_id: str | None = None,
    persona_id: str | None = None,
) -> None:
    session = _get_or_create_current_session()
    updates: dict[str, object] = {}
    if server is not None:
        updates["server"] = server
    if model is not None:
        updates["model"] = model
    if preset_id is not None:
        updates["preset_id"] = preset_id
    if persona_id is not None:
        updates["persona_id"] = persona_id
    if updates:
        session_service.update_session(str(session["id"]), updates)


def rename_chat_session(session_id: str | None, title: str) -> tuple[dict, str]:
    if not session_id:
        choices, current_id = list_chat_session_choices()
        return gr.update(choices=choices, value=current_id), "No session selected."

    try:
        renamed = session_service.rename_session(session_id, title)
    except ValueError as exc:
        choices, current_id = list_chat_session_choices()
        return gr.update(choices=choices, value=current_id), str(exc)

    choices, current_id = list_chat_session_choices()
    if renamed is None:
        return gr.update(choices=choices, value=current_id), "Session not found."
    return gr.update(choices=choices, value=current_id), "Chat renamed."


def rename_chat_session_from_state(session_id: str | None, title: str) -> tuple[dict, str | None, str]:
    if not session_id:
        dataset_update, current_id = _session_dataset_update()
        return dataset_update, current_id, "No session selected."

    try:
        renamed = session_service.rename_session(session_id, title)
    except ValueError as exc:
        dataset_update, current_id = _session_dataset_update()
        return dataset_update, current_id, str(exc)

    dataset_update, current_id = _session_dataset_update()
    if renamed is None:
        return dataset_update, current_id, "Session not found."
    return dataset_update, current_id, "Chat renamed."


def delete_chat_session(session_id: str | None) -> tuple[dict, list[dict[str, str]], list[dict[str, str]], str]:
    if not session_id:
        choices, current_id = list_chat_session_choices()
        history = load_current_chat_history()
        return gr.update(choices=choices, value=current_id), history, history, "No session selected."

    deleted = session_service.delete_session(session_id)
    choices, current_id = list_chat_session_choices()
    history = load_current_chat_history()
    if not deleted:
        return gr.update(choices=choices, value=current_id), history, history, "Session not found."
    return gr.update(choices=choices, value=current_id), history, history, "Chat deleted."


def delete_chat_session_with_state(
    session_id: str | None,
) -> tuple[
    dict,
    list[dict[str, str]],
    list[dict[str, str]],
    dict,
    str,
    dict,
    str,
    str,
    str,
    str | None,
    str | None,
    str | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    str,
]:
    deleted = False
    if session_id:
        deleted = session_service.delete_session(session_id)

    session_choices, current_session_id = list_chat_session_choices()
    session = _get_or_create_current_session()
    state = _resolve_session_ui_state(session)
    history = state["history"]
    status = "Chat deleted." if deleted else "Session not found." if session_id else "No session selected."
    return (
        gr.update(choices=session_choices, value=current_session_id),
        history,
        history,
        state["preset_dropdown"],
        state["preset_name"],
        state["persona_dropdown"],
        state["persona_name"],
        state["persona_description"],
        state["persona_system_prompt"],
        state["persona_default_model"],
        state["persona_default_preset"],
        state["model"],
        state["llm_temperature"],
        state["llm_max_tokens"],
        state["llm_top_p"],
        state["llm_typical_p"],
        state["llm_num_ctx"],
        state["llm_temperature"],
        state["llm_max_tokens"],
        state["llm_top_p"],
        state["llm_typical_p"],
        state["llm_num_ctx"],
        status,
    )


def delete_chat_session_with_dataset_state(
    session_id: str | None,
) -> tuple[
    dict,
    str | None,
    list[dict[str, str]],
    list[dict[str, str]],
    dict,
    str,
    dict,
    str,
    str,
    str,
    str | None,
    str | None,
    str | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    str,
]:
    deleted = False
    if session_id:
        deleted = session_service.delete_session(session_id)

    dataset_update, current_session_id = _session_dataset_update()
    session = _get_or_create_current_session()
    state = _resolve_session_ui_state(session)
    history = state["history"]
    status = "Chat deleted." if deleted else "Session not found." if session_id else "No session selected."
    return (
        dataset_update,
        current_session_id,
        history,
        history,
        state["preset_dropdown"],
        state["preset_name"],
        state["persona_dropdown"],
        state["persona_name"],
        state["persona_description"],
        state["persona_system_prompt"],
        state["persona_default_model"],
        state["persona_default_preset"],
        state["model"],
        state["llm_temperature"],
        state["llm_max_tokens"],
        state["llm_top_p"],
        state["llm_typical_p"],
        state["llm_num_ctx"],
        state["llm_temperature"],
        state["llm_max_tokens"],
        state["llm_top_p"],
        state["llm_typical_p"],
        state["llm_num_ctx"],
        status,
    )


def list_preset_choices() -> tuple[list[tuple[str, str]], str | None]:
    presets = preset_service.list_presets()
    choices = [(str(item.get("name", "Preset")), str(item["id"])) for item in presets]
    current = _get_or_create_current_session()
    current_preset_id = current.get("preset_id")
    if current_preset_id and any(value == current_preset_id for _label, value in choices):
        return choices, str(current_preset_id)
    return choices, None


def list_persona_choices() -> tuple[list[tuple[str, str]], str | None]:
    personas = persona_service.list_personas()
    choices = [(str(item.get("name", "Persona")), str(item["id"])) for item in personas]
    current = _get_or_create_current_session()
    current_persona_id = current.get("persona_id")
    if current_persona_id and any(value == current_persona_id for _label, value in choices):
        return choices, str(current_persona_id)
    return choices, None


def _resolve_session_ui_state(session: dict[str, object] | None) -> dict[str, object]:
    current = session or _get_or_create_current_session()
    preset_choices, _ = list_preset_choices()
    persona_choices, _ = list_persona_choices()

    persona_id = str(current.get("persona_id") or "") or None
    preset_id = str(current.get("preset_id") or "") or None
    model_value = str(current.get("model") or "") or None

    persona = persona_service.get_persona(persona_id) if persona_id else None
    preset = preset_service.get_preset(preset_id) if preset_id else None

    return {
        "history": _session_messages_to_history(current),
        "preset_dropdown": gr.update(choices=preset_choices, value=preset_id),
        "preset_name": str(preset.get("name", "")) if preset else "",
        "persona_dropdown": gr.update(choices=persona_choices, value=persona_id),
        "persona_name": str(persona.get("name", "")) if persona else "",
        "persona_description": str(persona.get("description", "")) if persona else "",
        "persona_system_prompt": str(persona.get("system_prompt", "")) if persona else "",
        "persona_default_model": (str(persona.get("default_model", "")) or None) if persona else None,
        "persona_default_preset": (str(persona.get("default_preset", "")) or None) if persona else None,
        "model": model_value,
        "llm_temperature": preset["llm_temperature"] if preset else gr.update(),
        "llm_max_tokens": preset["llm_max_tokens"] if preset else gr.update(),
        "llm_top_p": preset["llm_top_p"] if preset else gr.update(),
        "llm_typical_p": preset["llm_typical_p"] if preset else gr.update(),
        "llm_num_ctx": preset["llm_num_ctx"] if preset else gr.update(),
    }


def apply_preset_to_current_session(
    preset_id: str | None,
) -> tuple[dict, str, float | int | None, float | int | None, float | int | None, float | int | None, float | int | None, float | int | None, float | int | None, float | int | None, float | int | None, float | int | None, str]:
    choices, current_value = list_preset_choices()
    if not preset_id:
        session = _get_or_create_current_session()
        session_service.update_session(str(session["id"]), {"preset_id": None})
        return (
            gr.update(choices=choices, value=None),
            "",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "Preset cleared for current chat.",
        )

    preset = preset_service.get_preset(preset_id)
    if preset is None:
        return (
            gr.update(choices=choices, value=current_value),
            "",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "Preset not found.",
        )

    session = _get_or_create_current_session()
    session_service.update_session(str(session["id"]), {"preset_id": preset_id})
    return (
        gr.update(choices=choices, value=preset_id),
        str(preset.get("name", "")),
        preset["llm_temperature"],
        preset["llm_max_tokens"],
        preset["llm_top_p"],
        preset["llm_typical_p"],
        preset["llm_num_ctx"],
        preset["llm_temperature"],
        preset["llm_max_tokens"],
        preset["llm_top_p"],
        preset["llm_typical_p"],
        preset["llm_num_ctx"],
        f"Applied preset: {preset['name']}",
    )


def save_preset_from_values(
    preset_id: str | None,
    name: str,
    temperature: float,
    max_tokens: float,
    top_p: float,
    typical_p: float,
    num_ctx: float,
) -> tuple[dict, dict, str]:
    payload = {
        "name": name.strip() or "Custom Preset",
        "llm_temperature": float(temperature),
        "llm_max_tokens": int(max_tokens),
        "llm_top_p": float(top_p),
        "llm_typical_p": float(typical_p),
        "llm_num_ctx": int(num_ctx),
    }

    target = preset_service.get_preset(preset_id) if preset_id else None
    if target and not target.get("builtin"):
        saved = preset_service.update_preset(
            preset_id,
            {
                "name": payload["name"],
                "llm_temperature": payload["llm_temperature"],
                "llm_max_tokens": payload["llm_max_tokens"],
                "llm_top_p": payload["llm_top_p"],
                "llm_typical_p": payload["llm_typical_p"],
                "llm_num_ctx": payload["llm_num_ctx"],
            },
        )
        status = "Preset updated." if saved else "Preset not found."
        selected = preset_id
    else:
        saved = preset_service.create_preset(payload["name"], payload)
        status = "Preset created."
        selected = str(saved["id"])

    session = _get_or_create_current_session()
    session_service.update_session(str(session["id"]), {"preset_id": selected})
    choices, _current = list_preset_choices()
    update = gr.update(choices=choices, value=selected)
    return update, update, status


def delete_selected_preset(preset_id: str | None) -> tuple[dict, dict, str]:
    if not preset_id:
        choices, current = list_preset_choices()
        update = gr.update(choices=choices, value=current)
        return update, update, "No preset selected."

    deleted = preset_service.delete_preset(preset_id)
    session = _get_or_create_current_session()
    current_updates: dict[str, object] = {}
    if session.get("preset_id") == preset_id:
        current_updates["preset_id"] = None
    if current_updates:
        session_service.update_session(str(session["id"]), current_updates)
    choices, current = list_preset_choices()
    if not deleted:
        update = gr.update(choices=choices, value=current)
        return update, update, "Builtin presets cannot be deleted."
    update = gr.update(choices=choices, value=current)
    return update, update, "Preset deleted."


def load_selected_persona(
    persona_id: str | None,
) -> tuple[
    dict,
    str,
    str,
    str,
    str | None,
    str | None,
    dict,
    str | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    float | int | None,
    str,
]:
    choices, current = list_persona_choices()
    if not persona_id:
        session = _get_or_create_current_session()
        session_service.update_session(str(session["id"]), {"persona_id": None})
        return (
            gr.update(choices=choices, value=None),
            "",
            "",
            "",
            None,
            None,
            gr.update(),
            None,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "Persona cleared for current chat.",
        )

    persona = persona_service.get_persona(persona_id)
    if persona is None:
        return (
            gr.update(choices=choices, value=current),
            "",
            "",
            "",
            None,
            None,
            gr.update(),
            None,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "Persona not found.",
        )

    session = _get_or_create_current_session()
    updates: dict[str, object] = {"persona_id": persona_id}
    preset_id = str(persona.get("default_preset") or "") or None
    model_value = str(persona.get("default_model") or "") or None
    preset = preset_service.get_preset(preset_id) if preset_id else None
    if preset_id:
        updates["preset_id"] = preset_id
    if model_value:
        updates["model"] = model_value
    session_service.update_session(str(session["id"]), updates)
    return (
        gr.update(choices=choices, value=persona_id),
        str(persona.get("name", "")),
        str(persona.get("description", "")),
        str(persona.get("system_prompt", "")),
        model_value,
        preset_id,
        gr.update(value=preset_id),
        model_value,
        preset["llm_temperature"] if preset else gr.update(),
        preset["llm_max_tokens"] if preset else gr.update(),
        preset["llm_top_p"] if preset else gr.update(),
        preset["llm_typical_p"] if preset else gr.update(),
        preset["llm_num_ctx"] if preset else gr.update(),
        preset["llm_temperature"] if preset else gr.update(),
        preset["llm_max_tokens"] if preset else gr.update(),
        preset["llm_top_p"] if preset else gr.update(),
        preset["llm_typical_p"] if preset else gr.update(),
        preset["llm_num_ctx"] if preset else gr.update(),
        f"Applied persona: {persona.get('name', 'Persona')}",
    )


def save_persona(
    persona_id: str | None,
    name: str,
    description: str,
    system_prompt: str,
    default_model: str | None,
    default_preset: str | None,
) -> tuple[dict, str]:
    if persona_id:
        saved = persona_service.update_persona(
            persona_id,
            {
                "name": name.strip() or "New Persona",
                "description": description.strip(),
                "system_prompt": system_prompt,
                "default_model": default_model,
                "default_preset": default_preset,
            },
        )
        selected = persona_id if saved else None
        status = "Persona updated." if saved else "Persona not found."
    else:
        saved = persona_service.create_persona(
            name.strip() or "New Persona",
            system_prompt,
            description=description,
            default_model=default_model,
            default_preset=default_preset,
        )
        selected = str(saved["id"])
        status = "Persona created."

    session = _get_or_create_current_session()
    if selected:
        session_service.update_session(str(session["id"]), {"persona_id": selected})
    choices, _current = list_persona_choices()
    return gr.update(choices=choices, value=selected), status


def delete_selected_persona(persona_id: str | None) -> tuple[dict, str, str, str, str | None, str | None, str]:
    if not persona_id:
        choices, current = list_persona_choices()
        return gr.update(choices=choices, value=current), "", "", "", None, None, "No persona selected."

    deleted = persona_service.delete_persona(persona_id)
    session = _get_or_create_current_session()
    if session.get("persona_id") == persona_id:
        session_service.update_session(str(session["id"]), {"persona_id": None})
    choices, current = list_persona_choices()
    if not deleted:
        return gr.update(choices=choices, value=current), "", "", "", None, None, "Persona not found."
    return gr.update(choices=choices, value=current), "", "", "", None, None, "Persona deleted."


def list_prompt_choices() -> tuple[list[tuple[str, str]], str | None]:
    prompts = prompt_service.list_prompts()
    choices = [
        (f"{item['name']} [{item['category']}]" + (" *" if item.get("favorite") else ""), str(item["id"]))
        for item in prompts
    ]
    return choices, choices[0][1] if choices else None


def load_selected_prompt(prompt_id: str | None) -> tuple[dict, str, str, str, bool, str]:
    choices, current = list_prompt_choices()
    if not prompt_id:
        return gr.update(choices=choices, value=current), "", "", "", False, "No prompt selected."

    prompt = prompt_service.get_prompt(prompt_id)
    if prompt is None:
        return gr.update(choices=choices, value=current), "", "", "", False, "Prompt not found."

    return (
        gr.update(choices=choices, value=prompt_id),
        str(prompt.get("name", "")),
        str(prompt.get("category", "")),
        str(prompt.get("content", "")),
        bool(prompt.get("favorite", False)),
        "Prompt loaded.",
    )


def save_prompt_entry(
    prompt_id: str | None,
    name: str,
    category: str,
    content: str,
    favorite: bool,
) -> tuple[dict, str]:
    if prompt_id:
        saved = prompt_service.update_prompt(
            prompt_id,
            {
                "name": name.strip() or "Untitled Prompt",
                "category": category.strip() or "general",
                "content": content,
                "favorite": bool(favorite),
            },
        )
        selected = prompt_id if saved else None
        status = "Prompt updated." if saved else "Prompt not found."
    else:
        saved = prompt_service.create_prompt(
            name.strip() or "Untitled Prompt",
            content,
            category=category.strip() or "general",
            favorite=bool(favorite),
        )
        selected = str(saved["id"])
        status = "Prompt created."

    choices, _current = list_prompt_choices()
    return gr.update(choices=choices, value=selected), status


def delete_selected_prompt(prompt_id: str | None) -> tuple[dict, str, str, str, bool, str]:
    if not prompt_id:
        choices, current = list_prompt_choices()
        return gr.update(choices=choices, value=current), "", "", "", False, "No prompt selected."

    deleted = prompt_service.delete_prompt(prompt_id)
    choices, current = list_prompt_choices()
    if not deleted:
        return gr.update(choices=choices, value=current), "", "", "", False, "Prompt not found."
    return gr.update(choices=choices, value=current), "", "", "", False, "Prompt deleted."


def insert_selected_prompt_into_workspace(prompt_id: str | None, current_text: str) -> tuple[str, str]:
    if not prompt_id:
        return current_text, "No prompt selected."

    prompt = prompt_service.get_prompt(prompt_id)
    if prompt is None:
        return current_text, "Prompt not found."

    content = str(prompt.get("content", ""))
    if not current_text.strip():
        return content, "Prompt inserted."
    return current_text.rstrip() + "\n\n" + content, "Prompt inserted."


def _slugify_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "chat"


def export_current_chat_markdown() -> tuple[str, str]:
    session = _get_or_create_current_session()
    messages = session.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return "", "Current chat is empty."

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    title = str(session.get("title", "chat")).strip() or "chat"
    timestamp = str(session.get("updated_at", "")).replace(":", "-")
    filename = f"{_slugify_filename(title)}-{_slugify_filename(timestamp)}.md"
    path = EXPORTS_DIR / filename

    lines = [f"# {title}", ""]
    if session.get("created_at"):
        lines.append(f"- Created: {session['created_at']}")
    if session.get("updated_at"):
        lines.append(f"- Updated: {session['updated_at']}")
    if session.get("model"):
        lines.append(f"- Model: {session['model']}")
    if session.get("server"):
        lines.append(f"- Server: {session['server']}")
    lines.append("")

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "assistant")).strip().capitalize()
        content = str(message.get("content", "")).rstrip()
        created_at = str(message.get("created_at", "")).strip()
        lines.append(f"## {role}")
        if created_at:
            lines.append(f"_Time: {created_at}_")
            lines.append("")
        lines.append(content)
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(path), f"Exported chat to {path.name}"

def _persist_user_message(session_id: str, user_text: str) -> None:
    if user_text.strip():
        session_service.append_message(session_id, "user", user_text)


def _persist_assistant_message(session_id: str, content: str) -> None:
    normalized = content.strip()
    if normalized:
        session_service.append_message(session_id, "assistant", normalized)

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
    ensure_not_stopped()
    keywords = _extract_search_keywords(question_text)
    try:
        num_results = int(search_num_results or 5)
    except (TypeError, ValueError):
        num_results = 5
    num_results = max(1, min(num_results, 20))

    result = tool_registry.execute(
        "web_search",
        {
            "query": keywords,
            "num_results": num_results,
            "provider": search_provider or "serper.dev",
        },
    )
    ensure_not_stopped()

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
    ensure_not_stopped()
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


def _try_auto_tool_intent(
    text: str | None,
    search_provider: str | None,
    search_num_results: int | float | None,
    search_summary_length: str | None,
) -> AutoToolAction | None:
    return auto_tool_planner.plan(
        text=text,
        search_provider=search_provider,
        search_num_results=search_num_results,
        search_summary_length=search_summary_length,
    )


def _build_history_prompt(history: list[dict[str, str]], user_text: str, system_prompt: str | None = None) -> str:
    prompt = ""
    if system_prompt and system_prompt.strip():
        prompt += f"System: {system_prompt.strip()}\n\n"
    prompt += "Previous conversation:\n" if history else ""
    for msg in history:
        prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    prompt += f"\nUser: {user_text}\n"
    return prompt


def _build_ollama_options(
    llm_temperature,
    llm_max_tokens,
    llm_top_p,
    llm_typical_p,
    llm_num_ctx,
) -> dict[str, float | int]:
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
    try:
        if llm_top_p is not None:
            options["top_p"] = float(llm_top_p)
    except (TypeError, ValueError):
        pass
    try:
        if llm_typical_p is not None:
            options["typical_p"] = float(llm_typical_p)
    except (TypeError, ValueError):
        pass
    try:
        if llm_num_ctx is not None:
            options["num_ctx"] = int(llm_num_ctx)
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
    return model_runtime.post_chat_once(url, headers, payload)


def _build_fallback_payloads(
    model: str,
    prompt: str,
    user_text: str,
    options: dict[str, float | int],
    image_payload: dict[str, object],
) -> list[tuple[str, dict[str, object]]]:
    payloads: list[tuple[str, dict[str, object]]] = []

    full_message: dict[str, object] = {"role": "user", "content": prompt}
    if image_payload:
        full_message.update(image_payload)

    user_only_message: dict[str, object] = {"role": "user", "content": user_text or prompt}
    if image_payload:
        user_only_message.update(image_payload)

    payloads.append(
        (
            "full_prompt_with_options",
            {
                "model": model,
                "messages": [full_message],
                "options": options,
                "stream": True,
            },
        )
    )
    payloads.append(
        (
            "full_prompt_without_options",
            {
                "model": model,
                "messages": [full_message],
                "stream": True,
            },
        )
    )
    payloads.append(
        (
            "user_only_with_options",
            {
                "model": model,
                "messages": [user_only_message],
                "options": options,
                "stream": True,
            },
        )
    )
    payloads.append(
        (
            "user_only_without_options",
            {
                "model": model,
                "messages": [user_only_message],
                "stream": True,
            },
        )
    )

    return payloads


def _stream_chat(url: str, headers: dict[str, str], payload: dict):
    yield from model_runtime.stream_chat(url, headers, payload)


def ask_question_stream(
    question,
    history,
    model,
    selected_server,
    llm_temperature,
    llm_max_tokens,
    llm_top_p,
    llm_typical_p,
    llm_num_ctx,
    search_provider,
    web_search_enabled,
    search_num_results,
    search_summary_length,
):
    clear_stop()
    ensure_not_stopped()

    current_session = _get_or_create_current_session()
    session_id = str(current_session["id"])
    history = _session_messages_to_history(current_session)
    effective_model = model
    system_prompt = ""

    persona_id = current_session.get("persona_id")
    persona_default_preset_id: str | None = None
    if isinstance(persona_id, str) and persona_id:
        persona = persona_service.get_persona(persona_id)
        if persona is not None:
            system_prompt = str(persona.get("system_prompt", "")).strip()
            if persona.get("default_model"):
                effective_model = str(persona["default_model"])
            if persona.get("default_preset"):
                persona_default_preset_id = str(persona["default_preset"])

    effective_temperature = llm_temperature
    effective_max_tokens = llm_max_tokens
    effective_top_p = llm_top_p
    effective_typical_p = llm_typical_p
    effective_num_ctx = llm_num_ctx

    preset_id = current_session.get("preset_id")
    if (not isinstance(preset_id, str) or not preset_id) and persona_default_preset_id:
        preset_id = persona_default_preset_id
    if isinstance(preset_id, str) and preset_id:
        preset_values = preset_service.apply_preset(
            preset_id,
            {
                "llm_temperature": llm_temperature,
                "llm_max_tokens": llm_max_tokens,
                "llm_top_p": llm_top_p,
                "llm_typical_p": llm_typical_p,
                "llm_num_ctx": llm_num_ctx,
            },
        )
        if preset_values is not None:
            effective_temperature = preset_values["llm_temperature"]
            effective_max_tokens = preset_values["llm_max_tokens"]
            effective_top_p = preset_values["llm_top_p"]
            effective_typical_p = preset_values["llm_typical_p"]
            effective_num_ctx = preset_values["llm_num_ctx"]

    user_text = question.get("text") or ""
    web_refs: list[dict[str, str]] = []

    new_messages = history.copy()
    new_messages.append({"role": "user", "content": user_text})

    assistant_message = {"role": "assistant", "content": ""}
    new_messages.append(assistant_message)

    # 1) Manual tool commands (existing behavior retained)
    _persist_user_message(session_id, user_text)
    tool_output = _try_execute_tool_command(user_text, search_provider=search_provider)
    if tool_output is not None:
        assistant_message["content"] = tool_output
        _persist_assistant_message(session_id, assistant_message["content"])
        yield new_messages[-1], "Tool command executed"
        return

    auto_tool_action = _try_auto_tool_intent(
        user_text,
        search_provider=search_provider,
        search_num_results=search_num_results,
        search_summary_length=search_summary_length,
    )
    if isinstance(auto_tool_action, DirectAction):
        assistant_message["content"] = auto_tool_action.content.strip() or "Tool executed."
        _persist_assistant_message(session_id, assistant_message["content"])
        yield new_messages[-1], "Tool intent executed"
        return

    session_service.update_session(
        session_id,
        {
            "model": effective_model,
            "server": selected_server,
            "preset_id": preset_id,
        },
    )

    if effective_model is None:
        if isinstance(auto_tool_action, FetchForLlmAction):
            lines = [f"已抓取網址：{auto_tool_action.url}", f"狀態碼：{auto_tool_action.status}"]
            if auto_tool_action.title:
                lines.append(f"標題：{auto_tool_action.title}")
            lines.append(f"內容摘要：{auto_tool_action.fallback_summary}")
            assistant_message["content"] = "\n".join(lines)
            yield new_messages[-1], "Tool intent executed (fallback summary)"
            return
        if isinstance(auto_tool_action, SearchForLlmAction):
            assistant_message["content"] = auto_tool_action.fallback_content.strip() or "搜尋完成。"
            yield new_messages[-1], "Tool intent executed (fallback summary)"
            return
        assistant_message["content"] = "Please select a model."
        yield new_messages[-1], "Missing model"
        return

    if selected_server is None:
        if isinstance(auto_tool_action, FetchForLlmAction):
            lines = [f"已抓取網址：{auto_tool_action.url}", f"狀態碼：{auto_tool_action.status}"]
            if auto_tool_action.title:
                lines.append(f"標題：{auto_tool_action.title}")
            lines.append(f"內容摘要：{auto_tool_action.fallback_summary}")
            assistant_message["content"] = "\n".join(lines)
            yield new_messages[-1], "Tool intent executed (fallback summary)"
            return
        if isinstance(auto_tool_action, SearchForLlmAction):
            assistant_message["content"] = auto_tool_action.fallback_content.strip() or "搜尋完成。"
            yield new_messages[-1], "Tool intent executed (fallback summary)"
            return
        assistant_message["content"] = "Please select a server."
        yield new_messages[-1], "Missing server"
        return

    try:
        ensure_not_stopped()
        url = _build_chat_url(str(selected_server))
        headers = {"Content-Type": "application/json"}
        options = _build_ollama_options(
            effective_temperature,
            effective_max_tokens,
            effective_top_p,
            effective_typical_p,
            effective_num_ctx,
        )

        prompt = _build_history_prompt(history, user_text, system_prompt)

        if isinstance(auto_tool_action, FetchForLlmAction):
            fetched_url = auto_tool_action.url
            fetched_status = auto_tool_action.status
            fetched_title = auto_tool_action.title
            fetched_description = auto_tool_action.description
            fetched_text = auto_tool_action.text
            fallback_summary = auto_tool_action.fallback_summary

            summary_prompt = build_fetch_summary_prompt(
                user_request=user_text,
                fetched_url=fetched_url,
                fetched_status=fetched_status,
                fetched_title=fetched_title,
                fetched_description=fetched_description,
                fetched_text=fetched_text,
            )

            summary_payload = {
                "model": effective_model,
                "messages": [{"role": "user", "content": summary_prompt}],
                "options": options,
                "stream": True,
            }

            yield new_messages[-1], "Summarizing fetched content"
            try:
                for chunk in _stream_chat(url, headers, summary_payload):
                    ensure_not_stopped()
                    assistant_message["content"] += chunk
                    new_messages[-1]["content"] = assistant_message["content"]
                    yield new_messages[-1], "Summarizing fetched content"
                    time.sleep(0.02)
            except requests.RequestException as exc:
                logging.warning("LLM summary for fetched URL failed: %s", exc)

            if not assistant_message["content"].strip():
                retry_payload = dict(summary_payload)
                retry_payload["stream"] = False
                try:
                    retry_text = _post_chat_once(url, headers, retry_payload).strip()
                    if retry_text:
                        assistant_message["content"] = retry_text
                        new_messages[-1]["content"] = assistant_message["content"]
                except requests.RequestException as exc:
                    logging.warning("Retry LLM summary for fetched URL failed: %s", exc)

            if not assistant_message["content"].strip():
                logging.info("Using fallback summary for fetched URL.")
                lines = [f"已抓取網址：{fetched_url}", f"狀態碼：{fetched_status}"]
                if fetched_title:
                    lines.append(f"標題：{fetched_title}")
                lines.append(f"內容摘要：{fallback_summary}")
                assistant_message["content"] = "\n".join(lines)
                new_messages[-1]["content"] = assistant_message["content"]

            yield new_messages[-1], "Completed"
            _persist_assistant_message(session_id, assistant_message["content"])
            return

        if isinstance(auto_tool_action, SearchForLlmAction):
            search_provider_name = auto_tool_action.provider or str(search_provider or "serper.dev").strip()
            search_query = auto_tool_action.query or user_text
            summary_length = normalize_summary_length(auto_tool_action.summary_length or str(search_summary_length))
            style_cfg = summary_style_config(summary_length)
            points_min = int(style_cfg["points_min"])
            points_max = int(style_cfg["points_max"])
            prompt_hint = str(style_cfg["prompt_hint"])
            search_results = auto_tool_action.results
            fallback_content = auto_tool_action.fallback_content.strip()

            if isinstance(search_results, list):
                compact_results = [
                    {
                        "title": str(item.get("title", "")).strip(),
                        "snippet": clean_search_snippet(str(item.get("snippet", "")), max_len=260),
                        "url": str(item.get("url", "")).strip(),
                    }
                    for item in search_results
                ]
            else:
                compact_results = []

            summary_prompt = build_search_summary_prompt(
                user_request=user_text,
                search_query=search_query,
                search_provider_name=search_provider_name,
                compact_results_json=json.dumps(compact_results, ensure_ascii=False, indent=2),
                points_min=points_min,
                points_max=points_max,
                prompt_hint=prompt_hint,
            )

            summary_payload = {
                "model": effective_model,
                "messages": [{"role": "user", "content": summary_prompt}],
                "options": options,
                "stream": True,
            }

            yield new_messages[-1], "Summarizing search results"
            try:
                for chunk in _stream_chat(url, headers, summary_payload):
                    ensure_not_stopped()
                    assistant_message["content"] += chunk
                    new_messages[-1]["content"] = assistant_message["content"]
                    yield new_messages[-1], "Summarizing search results"
                    time.sleep(0.02)
            except requests.RequestException as exc:
                logging.warning("LLM summary for web search failed: %s", exc)

            if not assistant_message["content"].strip():
                retry_payload = dict(summary_payload)
                retry_payload["stream"] = False
                try:
                    retry_text = _post_chat_once(url, headers, retry_payload).strip()
                    if retry_text:
                        assistant_message["content"] = retry_text
                        new_messages[-1]["content"] = assistant_message["content"]
                except requests.RequestException as exc:
                    logging.warning("Retry LLM summary for web search failed: %s", exc)

            if not assistant_message["content"].strip():
                logging.info("Using fallback summary for web search.")
                assistant_message["content"] = fallback_content or "搜尋完成。"
                new_messages[-1]["content"] = assistant_message["content"]

            yield new_messages[-1], "Completed"
            _persist_assistant_message(session_id, assistant_message["content"])
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
        for message, status in run_legacy_conversation_stream(
            question=question,
            history=history,
            user_text=user_text,
            model=effective_model,
            search_provider=search_provider,
            web_search_enabled=bool(web_search_enabled),
            search_num_results=search_num_results,
            url=url,
            headers=headers,
            options=options,
            assistant_message=assistant_message,
            new_messages=new_messages,
            ensure_not_stopped=ensure_not_stopped,
            build_web_context=_build_web_context,
            build_history_prompt=lambda convo_history, latest_user_text: _build_history_prompt(
                convo_history,
                latest_user_text,
                system_prompt,
            ),
            build_tool_instruction=_build_tool_instruction,
            build_fallback_payloads=_build_fallback_payloads,
            stream_chat=_stream_chat,
            post_chat_once=_post_chat_once,
            linkify_reference_markers=_linkify_reference_markers,
            append_source_links=_append_source_links,
            tool_router=tool_router,
            image_payload=image_payload,
        ):
            yield message, status
        _persist_assistant_message(session_id, assistant_message["content"])
        return
    except requests.RequestException as exc:
        logging.error("Chat request failed: %s", exc)
        assistant_message["content"] = f"Server request failed: {exc}"
        _persist_assistant_message(session_id, assistant_message["content"])
        yield new_messages[-1], f"Error: {exc}"
    except OperationCancelled:
        assistant_message["content"] = "已停止回答。"
        new_messages[-1]["content"] = assistant_message["content"]
        yield new_messages[-1], "Stopped"
    except Exception as exc:  # noqa: BLE001
        logging.error("Unexpected chat error: %s", exc)
        assistant_message["content"] = "Unexpected error."
        _persist_assistant_message(session_id, assistant_message["content"])
        yield new_messages[-1], f"Error: {exc}"


def stop_response():
    request_stop()
    return gr.update(interactive=True), "Stopping..."


def send_prompt(prompt):
    return prompt


