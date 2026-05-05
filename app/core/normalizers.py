from __future__ import annotations

from typing import Any


def clamp_float(value: Any, minimum: float, maximum: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def normalize_provider(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"serper", "serper.dev", "serper_dev"}:
        return "serper.dev"
    if v == "tavily":
        return "tavily"
    return "serper.dev"


def normalize_summary_length(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"short", "medium", "long"}:
        return v
    return "medium"


def normalize_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "on"}:
            return True
        if v in {"false", "0", "no", "off"}:
            return False
    return default
