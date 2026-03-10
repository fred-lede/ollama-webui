from __future__ import annotations

from datetime import datetime
from typing import Any

from app.tools.base import ToolResult

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


class DateTimeTool:
    name = "datetime"
    description = (
        "Get current date/time. args: {\"timezone\": \"Asia/Taipei\", "
        "\"format\": \"%Y-%m-%d %H:%M:%S\"}"
    )

    def run(self, args: dict[str, Any]) -> ToolResult:
        tz_name = str(args.get("timezone", "")).strip()
        fmt = str(args.get("format", "%Y-%m-%d %H:%M:%S")).strip() or "%Y-%m-%d %H:%M:%S"

        try:
            if tz_name:
                if ZoneInfo is None:
                    return ToolResult(success=False, error="ZoneInfo is not available in this Python runtime")
                now = datetime.now(ZoneInfo(tz_name))
            else:
                now = datetime.now()

            return ToolResult(
                success=True,
                data={
                    "timezone": tz_name or "local",
                    "iso": now.isoformat(),
                    "formatted": now.strftime(fmt),
                },
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"Datetime tool failed: {exc}")
