from __future__ import annotations

from datetime import datetime
from typing import Any

from app.tools.base import BaseTool

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


class DateTimeTool(BaseTool):
    name = "datetime"
    description = "Get current local datetime. arguments: {\"timezone\":\"Asia/Taipei\",\"format\":\"%Y-%m-%d %H:%M:%S\"}"

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        tz_name = str(arguments.get("timezone", "")).strip()
        fmt = str(arguments.get("format", "%Y-%m-%d %H:%M:%S")).strip() or "%Y-%m-%d %H:%M:%S"

        if tz_name:
            if ZoneInfo is None:
                raise ValueError("ZoneInfo is not available in this Python runtime")
            now = datetime.now(ZoneInfo(tz_name))
        else:
            now = datetime.now()

        return {
            "timezone": tz_name or "local",
            "iso": now.isoformat(),
            "formatted": now.strftime(fmt),
        }
