from __future__ import annotations

import logging
import sys

import gradio as gr

from app.core.logging_setup import configure_logging
from app.ui.gradio_app import build_demo, css


def launch() -> None:
    configure_logging()
    try:
        demo = build_demo()
    except Exception:
        logging.exception("Failed to build UI — falling back to minimal demo")
        demo = gr.Blocks()
        with demo:
            gr.Markdown("# Startup Error\n\nCheck `log-webui.log` for details. Verify your JSON config files are valid.")

    try:
        demo.launch(show_error=True, inbrowser=True, theme=gr.themes.Default(), css=css)
    except Exception:
        logging.exception("Failed to launch Gradio server")
        print("Failed to launch Gradio server. Check log-webui.log for details.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    launch()
