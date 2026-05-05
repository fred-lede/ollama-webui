from __future__ import annotations

import re
from pathlib import Path

from app.ui.session_callbacks import _get_or_create_current_session

EXPORTS_DIR = Path(__file__).resolve().parents[2] / "exports"


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
