from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class JsonStore:
    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

    def ensure_dir(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir

    def path_for(self, filename: str) -> Path:
        return self.ensure_dir() / filename

    def ensure_file(self, filename: str, default_data: Any) -> Path:
        path = self.path_for(filename)
        if not path.exists():
            self.write(filename, default_data)
        return path

    def read(self, filename: str, default_data: Any) -> Any:
        path = self.ensure_file(filename, default_data)
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            payload = copy.deepcopy(default_data)
            self.write(filename, payload)
            return payload

    def write(self, filename: str, payload: Any) -> Path:
        path = self.path_for(filename)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path
