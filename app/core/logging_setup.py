from __future__ import annotations

import logging


def configure_logging() -> None:
    logging.basicConfig(
        filename="log-webui.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
        encoding="utf-8",
    )
