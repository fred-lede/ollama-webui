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
