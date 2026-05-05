from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.core.normalizers import clamp_float, clamp_int

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT_DIR / "server_settings.json"
LANGUAGE_FILE = ROOT_DIR / "language_settings.json"
DEFAULT_LLM_PARAMETERS: dict[str, float | int] = {
    "llm_temperature": 0.8,
    "llm_max_tokens": 2048,
    "llm_top_p": 0.9,
    "llm_typical_p": 0.7,
    "llm_num_ctx": 2048,
}


def load_language_settings() -> dict[str, Any]:
    if not LANGUAGE_FILE.exists():
        logging.info("%s not found, returning defaults.", LANGUAGE_FILE.name)
        return {"default_language": "English", "languages": {}}

    try:
        with LANGUAGE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {"default_language": "English", "languages": {}}
    except (OSError, json.JSONDecodeError) as exc:
        logging.error("Failed to load language settings: %s", exc)
        return {"default_language": "English", "languages": {}}


def save_default_language(default_language: str) -> bool:
    if not LANGUAGE_FILE.exists():
        return False

    try:
        with LANGUAGE_FILE.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    if not isinstance(payload, dict):
        payload = {}

    payload["default_language"] = str(default_language).strip() or "English"

    try:
        with LANGUAGE_FILE.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)
        return True
    except OSError:
        return False


def load_settings() -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not CONFIG_FILE.exists():
        logging.info("%s not found.", CONFIG_FILE.name)
        return [], None

    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            config = json.load(f)

        hosts = config.get("hosts", [])
        default_host = next((host for host in hosts if host.get("default")), None)
        logging.info("Loaded %s hosts.", len(hosts))
        return hosts, default_host
    except (OSError, json.JSONDecodeError) as exc:
        logging.error("Failed to load settings: %s", exc)
        return [], None


def save_settings(hosts: list[dict[str, Any]]) -> None:
    config: dict[str, Any] = {}

    # Keep any existing top-level keys (for example llm_parameters).
    if CONFIG_FILE.exists():
        try:
            with CONFIG_FILE.open("r", encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            config = {}

    config["hosts"] = hosts

    try:
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        logging.info("Saved settings with %s hosts.", len(hosts))
    except OSError as exc:
        logging.error("Failed to save settings: %s", exc)


def load_llm_parameters() -> dict[str, float | int]:
    data = dict(DEFAULT_LLM_PARAMETERS)
    if not CONFIG_FILE.exists():
        return data

    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return data

    params_list = config.get("llm_parameters", [])
    if not isinstance(params_list, list) or not params_list:
        return data
    first = params_list[0]
    if not isinstance(first, dict):
        return data

    def _to_float(name: str, default: float) -> float:
        try:
            return float(first.get(name, default))
        except (TypeError, ValueError):
            return default

    def _to_int(name: str, default: int) -> int:
        try:
            return int(first.get(name, default))
        except (TypeError, ValueError):
            return default

    data["llm_temperature"] = clamp_float(
        _to_float("llm_temperature", float(DEFAULT_LLM_PARAMETERS["llm_temperature"])),
        0.0,
        1.0,
        float(DEFAULT_LLM_PARAMETERS["llm_temperature"]),
    )
    data["llm_top_p"] = clamp_float(
        _to_float("llm_top_p", float(DEFAULT_LLM_PARAMETERS["llm_top_p"])),
        0.0,
        1.0,
        float(DEFAULT_LLM_PARAMETERS["llm_top_p"]),
    )
    data["llm_typical_p"] = clamp_float(
        _to_float("llm_typical_p", float(DEFAULT_LLM_PARAMETERS["llm_typical_p"])),
        0.0,
        1.0,
        float(DEFAULT_LLM_PARAMETERS["llm_typical_p"]),
    )
    data["llm_max_tokens"] = clamp_int(
        _to_int("llm_max_tokens", int(DEFAULT_LLM_PARAMETERS["llm_max_tokens"])),
        1,
        131072,
        int(DEFAULT_LLM_PARAMETERS["llm_max_tokens"]),
    )
    data["llm_num_ctx"] = clamp_int(
        _to_int("llm_num_ctx", int(DEFAULT_LLM_PARAMETERS["llm_num_ctx"])),
        1,
        131072,
        int(DEFAULT_LLM_PARAMETERS["llm_num_ctx"]),
    )
    return data


def save_llm_parameters(parameters: dict[str, Any]) -> bool:
    config: dict[str, Any] = {}
    if CONFIG_FILE.exists():
        try:
            with CONFIG_FILE.open("r", encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            config = {}

    payload = {
        "llm_temperature": clamp_float(
            parameters.get("llm_temperature", DEFAULT_LLM_PARAMETERS["llm_temperature"]),
            0.0,
            1.0,
            float(DEFAULT_LLM_PARAMETERS["llm_temperature"]),
        ),
        "llm_max_tokens": clamp_int(
            parameters.get("llm_max_tokens", DEFAULT_LLM_PARAMETERS["llm_max_tokens"]),
            1,
            131072,
            int(DEFAULT_LLM_PARAMETERS["llm_max_tokens"]),
        ),
        "llm_top_p": clamp_float(
            parameters.get("llm_top_p", DEFAULT_LLM_PARAMETERS["llm_top_p"]),
            0.0,
            1.0,
            float(DEFAULT_LLM_PARAMETERS["llm_top_p"]),
        ),
        "llm_typical_p": clamp_float(
            parameters.get("llm_typical_p", DEFAULT_LLM_PARAMETERS["llm_typical_p"]),
            0.0,
            1.0,
            float(DEFAULT_LLM_PARAMETERS["llm_typical_p"]),
        ),
        "llm_num_ctx": clamp_int(
            parameters.get("llm_num_ctx", DEFAULT_LLM_PARAMETERS["llm_num_ctx"]),
            1,
            131072,
            int(DEFAULT_LLM_PARAMETERS["llm_num_ctx"]),
        ),
    }
    config["llm_parameters"] = [payload]

    try:
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        logging.info("Saved LLM parameters.")
        return True
    except OSError as exc:
        logging.error("Failed to save LLM parameters: %s", exc)
        return False


# --- Search settings (migrated from app_settings.json) ---

_DEFAULT_SEARCH: dict[str, Any] = {
    "search": {
        "enabled": False,
        "provider": "serper.dev",
        "num_results": 5,
        "summary_length": "medium",
        "tavily_api_key": "",
        "serper_api_key": "",
        "tavily_api_url": "https://api.tavily.com/search",
        "serper_api_url": "https://google.serper.dev/search",
    }
}


def _migrate_legacy_app_settings() -> None:
    """One-time migration: merge app_settings.json search block into server_settings.json."""
    legacy = ROOT_DIR / "app_settings.json"
    if not legacy.exists():
        return
    try:
        with legacy.open("r", encoding="utf-8-sig") as f:
            old = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(old, dict):
        return
    search = old.get("search")
    if not isinstance(search, dict):
        return
    config: dict[str, Any] = {}
    if CONFIG_FILE.exists():
        try:
            with CONFIG_FILE.open("r", encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            config = {}
    config.setdefault("search", {}).update(search)
    try:
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        logging.info("Migrated search settings from app_settings.json to server_settings.json")
    except OSError:
        return


def load_search_settings() -> dict[str, Any]:
    _migrate_legacy_app_settings()
    merged = dict(_DEFAULT_SEARCH["search"])
    if CONFIG_FILE.exists():
        try:
            with CONFIG_FILE.open("r", encoding="utf-8") as f:
                config = json.load(f)
            saved = config.get("search", {})
            if isinstance(saved, dict):
                merged.update(saved)
        except (OSError, json.JSONDecodeError):
            pass
    return merged


def save_search_settings(search: dict[str, Any]) -> bool:
    config: dict[str, Any] = {}
    if CONFIG_FILE.exists():
        try:
            with CONFIG_FILE.open("r", encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            config = {}
    config["search"] = search
    try:
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        return True
    except OSError:
        return False
