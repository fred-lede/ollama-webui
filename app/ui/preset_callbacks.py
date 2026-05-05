from __future__ import annotations

from dataclasses import asdict, dataclass

import gradio as gr

from app.services.preset_service import PresetService
from app.ui.session_callbacks import _get_or_create_current_session

preset_service = PresetService()


@dataclass
class PresetUIState:
    preset_dropdown: object = None
    preset_name: str = ""
    llm_temperature: object = None
    llm_max_tokens: object = None
    llm_top_p: object = None
    llm_typical_p: object = None
    llm_num_ctx: object = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def list_preset_choices() -> tuple[list[tuple[str, str]], str | None]:
    presets = preset_service.list_presets()
    choices = [(str(item.get("name", "Preset")), str(item.get("id", ""))) for item in presets if item.get("id")]
    current = _get_or_create_current_session()
    current_preset_id = current.get("preset_id")
    if current_preset_id and any(value == current_preset_id for _label, value in choices):
        return choices, str(current_preset_id)
    return choices, None


def apply_preset_to_current_session(
    preset_id: str | None,
) -> tuple[PresetUIState, str]:
    choices, current_value = list_preset_choices()
    if not preset_id:
        session = _get_or_create_current_session()
        from app.ui.session_callbacks import session_service as _ss
        _ss.update_session(str(session.get("id", "")), {"preset_id": None})
        return PresetUIState(preset_dropdown=gr.update(choices=choices, value=None)), "Preset cleared for current chat."

    preset = preset_service.get_preset(preset_id)
    if preset is None:
        return PresetUIState(preset_dropdown=gr.update(choices=choices, value=current_value)), "Preset not found."

    session = _get_or_create_current_session()
    from app.ui.session_callbacks import session_service as _ss
    _ss.update_session(str(session.get("id", "")), {"preset_id": preset_id})
    return PresetUIState(
        preset_dropdown=gr.update(choices=choices, value=preset_id),
        preset_name=str(preset.get("name", "")),
        llm_temperature=preset.get("llm_temperature"),
        llm_max_tokens=preset.get("llm_max_tokens"),
        llm_top_p=preset.get("llm_top_p"),
        llm_typical_p=preset.get("llm_typical_p"),
        llm_num_ctx=preset.get("llm_num_ctx"),
    ), f"Applied preset: {preset.get('name', 'Preset')}"


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
    from app.ui.session_callbacks import session_service as _ss
    _ss.update_session(str(session.get("id", "")), {"preset_id": selected})
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
        from app.ui.session_callbacks import session_service as _ss
        _ss.update_session(str(session.get("id", "")), current_updates)
    choices, current = list_preset_choices()
    if not deleted:
        update = gr.update(choices=choices, value=current)
        return update, update, "Builtin presets cannot be deleted."
    update = gr.update(choices=choices, value=current)
    return update, update, "Preset deleted."
