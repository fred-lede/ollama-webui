from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

import gradio as gr

from app.services.session_service import SessionService

session_service = SessionService()
_SESSION_LABEL_LANGUAGE = "English"


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
    return f"{title} | {updated_at}" if updated_at else title


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
    choices = [(_build_session_choice_label(session), str(session.get("id", ""))) for session in sessions if session.get("id")]
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


@dataclass
class SessionUIState:
    history: list
    preset_dropdown: object = None
    preset_name: str = ""
    persona_dropdown: object = None
    persona_name: str = ""
    persona_description: str = ""
    persona_system_prompt: str = ""
    persona_default_model: str | None = None
    persona_default_preset: str | None = None
    model: str | None = None
    llm_temperature: object = None
    llm_max_tokens: object = None
    llm_top_p: object = None
    llm_typical_p: object = None
    llm_num_ctx: object = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _resolve_session_ui_state(session: dict[str, object] | None) -> SessionUIState:
    from app.ui.preset_callbacks import list_preset_choices
    from app.ui.persona_callbacks import list_persona_choices

    current = session or _get_or_create_current_session()
    preset_choices, _ = list_preset_choices()
    persona_choices, _ = list_persona_choices()

    persona_id = str(current.get("persona_id") or "") or None
    preset_id = str(current.get("preset_id") or "") or None
    model_value = str(current.get("model") or "") or None

    from app.ui.persona_callbacks import persona_service as _persona_service
    from app.ui.preset_callbacks import preset_service as _preset_service
    persona = _persona_service.get_persona(persona_id) if persona_id else None
    preset = _preset_service.get_preset(preset_id) if preset_id else None

    return SessionUIState(
        history=_session_messages_to_history(current),
        preset_dropdown=gr.update(choices=preset_choices, value=preset_id),
        preset_name=str(preset.get("name", "")) if preset else "",
        persona_dropdown=gr.update(choices=persona_choices, value=persona_id),
        persona_name=str(persona.get("name", "")) if persona else "",
        persona_description=str(persona.get("description", "")) if persona else "",
        persona_system_prompt=str(persona.get("system_prompt", "")) if persona else "",
        persona_default_model=(str(persona.get("default_model", "")) or None) if persona else None,
        persona_default_preset=(str(persona.get("default_preset", "")) or None) if persona else None,
        model=model_value,
        llm_temperature=preset.get("llm_temperature", gr.update()) if preset else gr.update(),
        llm_max_tokens=preset.get("llm_max_tokens", gr.update()) if preset else gr.update(),
        llm_top_p=preset.get("llm_top_p", gr.update()) if preset else gr.update(),
        llm_typical_p=preset.get("llm_typical_p", gr.update()) if preset else gr.update(),
        llm_num_ctx=preset.get("llm_num_ctx", gr.update()) if preset else gr.update(),
    )


def switch_chat_session_with_state(
    session_id: str | None,
) -> tuple[SessionUIState, str]:
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
    return state, status


def switch_chat_session_from_dataset(
    evt: gr.SelectData,
) -> tuple[str | None, SessionUIState, str]:
    session_id = _session_id_from_dataset_index(getattr(evt, "index", None))
    state, status = switch_chat_session_with_state(session_id)
    return session_id, state, status


def create_new_chat_session_with_choices() -> tuple[dict, list[dict[str, str]], list[dict[str, str]], str]:
    create_new_chat_session()
    choices, current_id = list_chat_session_choices()
    return gr.update(choices=choices, value=current_id), [], [], "Started a new chat."


def create_new_chat_session_with_state() -> tuple[dict, SessionUIState, str]:
    session = session_service.create_session()
    session_choices, current_session_id = list_chat_session_choices()
    state = _resolve_session_ui_state(session)
    return (
        gr.update(choices=session_choices, value=current_session_id),
        state,
        "Started a new chat.",
    )


def create_new_chat_session_with_dataset_state() -> tuple[dict, str | None, SessionUIState, str]:
    session = session_service.create_session()
    dataset_update, current_session_id = _session_dataset_update()
    state = _resolve_session_ui_state(session)
    return (
        dataset_update,
        current_session_id,
        state,
        "Started a new chat.",
    )


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
) -> tuple[dict, SessionUIState, str]:
    deleted = False
    if session_id:
        deleted = session_service.delete_session(session_id)

    session_choices, current_session_id = list_chat_session_choices()
    session = _get_or_create_current_session()
    state = _resolve_session_ui_state(session)
    status = "Chat deleted." if deleted else "Session not found." if session_id else "No session selected."
    return (
        gr.update(choices=session_choices, value=current_session_id),
        state,
        status,
    )


def delete_chat_session_with_dataset_state(
    session_id: str | None,
) -> tuple[dict, str | None, SessionUIState, str]:
    deleted = False
    if session_id:
        deleted = session_service.delete_session(session_id)

    dataset_update, current_session_id = _session_dataset_update()
    session = _get_or_create_current_session()
    state = _resolve_session_ui_state(session)
    status = "Chat deleted." if deleted else "Session not found." if session_id else "No session selected."
    return (
        dataset_update,
        current_session_id,
        state,
        status,
    )
