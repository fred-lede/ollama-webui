from __future__ import annotations

import gradio as gr

from app.core.logging_setup import configure_logging
from app.ui.gradio_app import build_demo, css


def launch() -> None:
    configure_logging()
    demo = build_demo()
    demo.launch(show_error=True, inbrowser=True, theme=gr.themes.Default(), css=css)


if __name__ == "__main__":
    launch()
