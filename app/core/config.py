from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT_DIR / "server_settings.json"
LANGUAGE_FILE = ROOT_DIR / "language_settings.json"


def load_language_settings() -> dict[str, Any]:
    if not LANGUAGE_FILE.exists():
        raise FileNotFoundError(f"{LANGUAGE_FILE} not found.")
    with LANGUAGE_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


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
