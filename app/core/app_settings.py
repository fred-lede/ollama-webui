from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
APP_SETTINGS_FILE = ROOT_DIR / "app_settings.json"


def load_app_settings() -> dict[str, Any]:
    if not APP_SETTINGS_FILE.exists():
        return {}

    try:
        with APP_SETTINGS_FILE.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_app_settings(settings: dict[str, Any]) -> bool:
    try:
        with APP_SETTINGS_FILE.open("w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        return True
    except OSError:
        return False
