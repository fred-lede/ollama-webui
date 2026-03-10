from __future__ import annotations

import json
import logging
import re
from urllib.parse import urlparse

import gradio as gr
import requests

from app.core.config import load_settings, save_settings


def is_valid_url(url: str) -> bool:
    # Accept IPv4 and domain names with http/https.
    regex = re.compile(r"^(http://|https://)([\w.-]+)(:\d+)?$")
    return re.match(regex, url) is not None


def fetch_models(server_address: str, server_port: int) -> list[str]:
    try:
        url = f"{server_address}:{server_port}/v1/models"
        logging.info("Fetching models from %s", url)
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        models_data = response.json().get("data", [])
        models = [model["id"] for model in models_data if "id" in model]
        logging.info("Models from %s:%s => %s", server_address, server_port, models)
        return models
    except requests.RequestException as exc:
        logging.error("Failed to fetch models from %s:%s: %s", server_address, server_port, exc)
        return []
    except json.JSONDecodeError as exc:
        logging.error("Model response JSON decode error: %s", exc)
        return []


def _normalize_server_base(selected_server: str) -> tuple[str, int]:
    raw = (selected_server or "").strip()
    if not raw:
        raise ValueError("請先選擇伺服器。")

    if "://" not in raw:
        raw = "http://" + raw

    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError("伺服器位址格式錯誤。")

    address = f"{parsed.scheme}://{parsed.hostname}"
    port = parsed.port or 11434
    return address, int(port)


def _status_light_html(level: str, label: str) -> str:
    palette = {
        "ok": ("#16a34a", "#ecfdf3"),
        "warn": ("#f59e0b", "#fff7ed"),
        "error": ("#dc2626", "#fef2f2"),
    }
    dot, bg = palette.get(level, palette["warn"])
    safe_label = str(label).strip() or "未測試"
    return (
        "<div style='display:inline-flex;align-items:center;gap:8px;"
        f"padding:6px 10px;border-radius:999px;background:{bg};'>"
        f"<span style='width:10px;height:10px;border-radius:50%;background:{dot};display:inline-block;'></span>"
        f"<span style='font-size:13px;'>{safe_label}</span>"
        "</div>"
    )


def test_llm_connection(selected_server: str | None, selected_model: str | None) -> tuple[str, str]:
    if not selected_server:
        return "請先選擇伺服器。", _status_light_html("warn", "未測試")
    if not selected_model:
        return "請先選擇模型。", _status_light_html("warn", "未測試")

    try:
        address, port = _normalize_server_base(str(selected_server))
        url = f"{address}:{port}/api/chat"
        payload = {
            "model": str(selected_model),
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "options": {"num_predict": 8, "temperature": 0},
        }

        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json() if response.content else {}
        has_message = isinstance(data, dict) and isinstance(data.get("message"), dict)
        if not has_message:
            logging.warning("LLM health check got unexpected shape: %s", data)

        return (
            f"LLM 連線正常：{selected_model}",
            _status_light_html("ok", "連線正常"),
        )
    except ValueError as exc:
        return str(exc), _status_light_html("warn", "設定不完整")
    except requests.RequestException as exc:
        logging.warning("LLM connection test failed: %s", exc)
        return (
            f"LLM 連線失敗：{exc}",
            _status_light_html("error", "連線失敗"),
        )
    except Exception as exc:  # noqa: BLE001
        logging.error("LLM connection test unexpected error: %s", exc)
        return (
            f"LLM 測試發生錯誤：{exc}",
            _status_light_html("error", "測試錯誤"),
        )


def handle_add_server(
    new_server_name: str,
    new_address: str,
    new_port: int,
    new_default_server: bool,
):
    logging.info(
        "Add server request: name=%s, address=%s, port=%s, default=%s",
        new_server_name,
        new_address,
        new_port,
        new_default_server,
    )

    if not is_valid_url(new_address):
        return (
            "Invalid server URL. Example: http://192.168.1.173",
            gr.update(),
            gr.update(),
            "",
            "",
        )

    hosts, _ = load_settings()

    for host in hosts:
        if host["address"] == new_address and host["port"] == new_port:
            return "Server already exists.", gr.update(), gr.update(), "", ""

    hosts.append(
        {
            "server_name": new_server_name,
            "address": new_address,
            "port": new_port,
            "default": new_default_server,
        }
    )

    if new_default_server:
        for host in hosts:
            host["default"] = host["address"] == new_address and host["port"] == new_port

    save_settings(hosts)

    server_choices = [(host["server_name"], f"{host['address']}:{host['port']}") for host in hosts]
    current_host = f"{new_address}:{new_port}"
    models = fetch_models(new_address, new_port)

    status = "Server added. Models refreshed." if models else "Server added, but failed to fetch models."

    return (
        status,
        gr.update(choices=server_choices, value=current_host),
        gr.update(choices=models, value=models[0] if models else None),
        "",
        "",
    )


def handle_server_change(selected_server: str | None):
    logging.info("Server changed: %s", selected_server)

    if not selected_server:
        return "Please select a server.", gr.update(choices=[], value=None), ""

    try:
        parsed_url = urlparse(selected_server)
        address = f"{parsed_url.scheme}://{parsed_url.hostname}"
        port = parsed_url.port or 11434
        models = fetch_models(address, port)

        status = "Server switched. Models refreshed." if models else "Failed to fetch models."

        return status, gr.update(choices=models, value=models[0] if models else None), selected_server
    except Exception as exc:  # noqa: BLE001
        logging.error("Server change failed: %s", exc)
        return "Server change failed.", gr.update(choices=[], value=None), ""
