from __future__ import annotations

import threading


class OperationCancelled(Exception):
    """Raised when the current operation is cancelled by user action."""


_STOP_EVENT = threading.Event()


def request_stop() -> None:
    _STOP_EVENT.set()


def clear_stop() -> None:
    _STOP_EVENT.clear()


def is_stop_requested() -> bool:
    return _STOP_EVENT.is_set()


def ensure_not_stopped() -> None:
    if _STOP_EVENT.is_set():
        raise OperationCancelled("Operation cancelled by user")
