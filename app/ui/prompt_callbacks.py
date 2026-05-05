from __future__ import annotations

import gradio as gr

from app.services.prompt_service import PromptService

prompt_service = PromptService()


def list_prompt_choices() -> tuple[list[tuple[str, str]], str | None]:
    prompts = prompt_service.list_prompts()
    choices = [
        (f"{item.get('name', 'Prompt')} [{item.get('category', 'general')}]" + (" *" if item.get("favorite") else ""), str(item.get("id", "")))
        for item in prompts
        if item.get("id")
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
