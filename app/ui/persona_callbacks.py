from __future__ import annotations

import gradio as gr

from app.services.persona_service import PersonaService
from app.ui.session_callbacks import SessionUIState, _get_or_create_current_session

persona_service = PersonaService()


def list_persona_choices() -> tuple[list[tuple[str, str]], str | None]:
    personas = persona_service.list_personas()
    choices = [(str(item.get("name", "Persona")), str(item.get("id", ""))) for item in personas if item.get("id")]
    current = _get_or_create_current_session()
    current_persona_id = current.get("persona_id")
    if current_persona_id and any(value == current_persona_id for _label, value in choices):
        return choices, str(current_persona_id)
    return choices, None


def load_selected_persona(
    persona_id: str | None,
) -> tuple[SessionUIState, str]:
    from app.ui.preset_callbacks import list_preset_choices

    choices, current = list_persona_choices()
    if not persona_id:
        session = _get_or_create_current_session()
        from app.ui.session_callbacks import session_service as _ss
        _ss.update_session(str(session.get("id", "")), {"persona_id": None})
        return SessionUIState(
            history=[],
            persona_dropdown=gr.update(choices=choices, value=None),
            preset_dropdown=gr.update(),
        ), "Persona cleared for current chat."

    persona = persona_service.get_persona(persona_id)
    if persona is None:
        return SessionUIState(
            history=[],
            persona_dropdown=gr.update(choices=choices, value=current),
            preset_dropdown=gr.update(),
        ), "Persona not found."

    from app.ui.preset_callbacks import preset_service as _preset_service

    session = _get_or_create_current_session()
    updates: dict[str, object] = {"persona_id": persona_id}
    preset_id = str(persona.get("default_preset") or "") or None
    model_value = str(persona.get("default_model") or "") or None
    preset = _preset_service.get_preset(preset_id) if preset_id else None
    if preset_id:
        updates["preset_id"] = preset_id
    if model_value:
        updates["model"] = model_value
    from app.ui.session_callbacks import session_service as _ss
    _ss.update_session(str(session.get("id", "")), updates)
    return SessionUIState(
        history=[],
        persona_dropdown=gr.update(choices=choices, value=persona_id),
        persona_name=str(persona.get("name", "")),
        persona_description=str(persona.get("description", "")),
        persona_system_prompt=str(persona.get("system_prompt", "")),
        persona_default_model=model_value,
        persona_default_preset=preset_id,
        preset_dropdown=gr.update(value=preset_id),
        model=model_value,
        llm_temperature=preset.get("llm_temperature", gr.update()) if preset else gr.update(),
        llm_max_tokens=preset.get("llm_max_tokens", gr.update()) if preset else gr.update(),
        llm_top_p=preset.get("llm_top_p", gr.update()) if preset else gr.update(),
        llm_typical_p=preset.get("llm_typical_p", gr.update()) if preset else gr.update(),
        llm_num_ctx=preset.get("llm_num_ctx", gr.update()) if preset else gr.update(),
    ), f"Applied persona: {persona.get('name', 'Persona')}"


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
        from app.ui.session_callbacks import session_service as _ss
        _ss.update_session(str(session.get("id", "")), {"persona_id": selected})
    choices, _current = list_persona_choices()
    return gr.update(choices=choices, value=selected), status


def delete_selected_persona(persona_id: str | None) -> tuple[dict, str, str, str, str | None, str | None, str]:
    if not persona_id:
        choices, current = list_persona_choices()
        return gr.update(choices=choices, value=current), "", "", "", None, None, "No persona selected."

    deleted = persona_service.delete_persona(persona_id)
    session = _get_or_create_current_session()
    if session.get("persona_id") == persona_id:
        from app.ui.session_callbacks import session_service as _ss
        _ss.update_session(str(session.get("id", "")), {"persona_id": None})
    choices, current = list_persona_choices()
    if not deleted:
        return gr.update(choices=choices, value=current), "", "", "", None, None, "Persona not found."
    return gr.update(choices=choices, value=current), "", "", "", None, None, "Persona deleted."
