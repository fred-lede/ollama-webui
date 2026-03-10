from __future__ import annotations

import gradio as gr

from app.core.app_settings import load_app_settings, save_app_settings
from app.core.config import load_language_settings, load_settings
from app.services.chat_service import ask_question_stream, send_prompt, stop_response
from app.services.server_service import fetch_models, handle_add_server, handle_server_change

css = """
.my-button {
    width: 110px !important;
}
.drawer-note {
    opacity: 0.85;
    font-size: 0.9rem;
}
"""


def _normalize_provider(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"serper", "serper.dev", "serper_dev"}:
        return "serper.dev"
    if v == "tavily":
        return "tavily"
    return "serper.dev"


def build_demo() -> gr.Blocks:
    language_settings = load_language_settings()
    current_language = language_settings.get("default_language", "English")
    translations = language_settings["languages"].get(current_language, {})

    settings = load_app_settings()
    search_cfg = settings.get("search", {}) if isinstance(settings.get("search", {}), dict) else {}
    default_search_provider = _normalize_provider(str(search_cfg.get("provider", "serper.dev")))
    default_tavily_api_key = str(search_cfg.get("tavily_api_key", ""))
    default_serper_api_key = str(search_cfg.get("serper_api_key", ""))
    default_search_num_results = int(search_cfg.get("num_results", 5) or 5)

    hosts, default_host = load_settings()
    server_choices = [(host["server_name"], f"{host['address']}:{host['port']}") for host in hosts]
    current_host = f"{default_host['address']}:{default_host['port']}" if default_host else None
    models = fetch_models(default_host["address"], default_host["port"]) if default_host else []

    def update_language(selected_language):
        nonlocal current_language, translations
        current_language = selected_language
        translations = language_settings["languages"].get(selected_language, {})
        return translations

    def toggle_web_search(enabled: bool):
        new_enabled = not bool(enabled)
        label = "Web Search: ON" if new_enabled else "Web Search: OFF"
        variant = "primary" if new_enabled else "secondary"
        return gr.update(value=new_enabled), gr.update(value=label, variant=variant)

    def toggle_settings_drawer(is_open: bool):
        new_open = not bool(is_open)
        label = "關閉設定 ✕" if new_open else "設定 ⚙"
        variant = "primary" if new_open else "secondary"
        return (
            gr.update(value=new_open),
            gr.update(visible=new_open),
            gr.update(value=label, variant=variant),
        )

    def save_search_settings_ui(provider: str, tavily_key: str, serper_key: str, num_results: float):
        existing = load_app_settings()
        existing_search = existing.get("search", {}) if isinstance(existing.get("search", {}), dict) else {}

        try:
            num = int(num_results)
        except (TypeError, ValueError):
            num = 5
        num = max(1, min(num, 10))

        existing["search"] = {
            "provider": _normalize_provider(provider),
            "num_results": num,
            "tavily_api_key": tavily_key.strip(),
            "serper_api_key": serper_key.strip(),
            "tavily_api_url": str(existing_search.get("tavily_api_url", "https://api.tavily.com/search")),
            "serper_api_url": str(existing_search.get("serper_api_url", "https://google.serper.dev/search")),
        }

        ok = save_app_settings(existing)
        if ok:
            return "搜尋設定已儲存。"
        return "搜尋設定儲存失敗。"

    with gr.Blocks() as demo:
        gr.Markdown("# Ollama WebUI")

        settings_open_state = gr.Checkbox(value=False, visible=False)
        search_provider_state = gr.State(default_search_provider)
        search_num_results_state = gr.State(default_search_num_results)

        with gr.Row(equal_height=False):
            with gr.Column(scale=9):
                with gr.Row():
                    server_dropdown = gr.Dropdown(
                        choices=server_choices,
                        allow_custom_value=True,
                        label=translations.get("select_server", "Select Server"),
                        value=current_host,
                        scale=3,
                    )
                    model_dropdown = gr.Dropdown(
                        choices=models,
                        allow_custom_value=True,
                        label=translations.get("select_model", "Select Model"),
                        value=models[0] if models else None,
                        scale=3,
                    )
                    web_search_enabled = gr.Checkbox(value=True, visible=False)
                    web_search_toggle = gr.Button(value="Web Search: ON", variant="primary", scale=2)
                    settings_toggle_button = gr.Button(value="設定 ⚙", variant="secondary", scale=2)

                chatbot = gr.Chatbot(
                    buttons=["copy", "copy_all"],
                    height=560,
                    autoscroll=True,
                )

                question_input = gr.MultimodalTextbox(
                    interactive=True,
                    autoscroll=True,
                    file_count="single",
                    placeholder=translations.get("enter_question", "Enter your question or upload image file with question..."),
                    show_label=False,
                    file_types=[".jpg", ".jpeg", ".png", ".bmp"],
                )

                with gr.Row():
                    stop_button = gr.Button(value=translations.get("stop_button", "Stop Answer"), elem_classes="my-button")
                    clean_answer_button = gr.ClearButton(
                        chatbot,
                        value=translations.get("clean_answer_button", "Clear Answer"),
                        elem_classes="my-button",
                    )

                status_output = gr.Textbox(
                    label=translations.get("status_display", "Status Display"),
                    show_label=True,
                    placeholder=translations.get(
                        "status_display_msg",
                        "Displaying the model's status or displaying an error message.",
                    ),
                    lines=1,
                )

                llm_temperature = gr.Slider(
                    0,
                    1,
                    step=0.1,
                    value=0.5,
                    label="Temperature",
                    info="Choose between 0 and 1",
                    interactive=True,
                    visible=False,
                )
                llm_max_tokens = gr.Number(label="Max Tokens", interactive=True, value=1500, visible=False)

                gr.ChatInterface(
                    fn=ask_question_stream,
                    multimodal=True,
                    additional_inputs=[
                        model_dropdown,
                        server_dropdown,
                        llm_temperature,
                        llm_max_tokens,
                        search_provider_state,
                        web_search_enabled,
                        search_num_results_state,
                    ],
                    fill_width=True,
                    fill_height=True,
                    chatbot=chatbot,
                    autofocus=False,
                    show_progress="full",
                    textbox=question_input,
                )

            with gr.Column(scale=4, visible=False) as settings_drawer:
                gr.Markdown("### 設定面板")
                gr.Markdown("<div class='drawer-note'>所有設定集中在這裡，儲存後即生效。</div>")

                with gr.Accordion("Search", open=True):
                    search_provider_dropdown = gr.Dropdown(
                        choices=[("Serper.dev", "serper.dev"), ("Tavily", "tavily")],
                        label="Search Provider",
                        value=default_search_provider,
                        interactive=True,
                    )
                    search_num_results_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=max(1, min(default_search_num_results, 10)),
                        label="Search Results Count",
                        info="How many web search results to use for answer grounding",
                        interactive=True,
                    )
                    tavily_api_key_input = gr.Textbox(
                        label="Tavily API Key",
                        value=default_tavily_api_key,
                        type="password",
                        placeholder="Paste Tavily API key",
                    )
                    serper_api_key_input = gr.Textbox(
                        label="Serper.dev API Key",
                        value=default_serper_api_key,
                        type="password",
                        placeholder="Paste Serper.dev API key",
                    )
                    save_search_setting_button = gr.Button(value="Save Search Settings", variant="secondary")

                with gr.Accordion("Server", open=False):
                    new_server_name_input = gr.Textbox(
                        label=translations.get("ai_server_name", "Ai Server Name"),
                        placeholder=translations.get(
                            "ai_server_name_msg",
                            "Ex: GMK-K9, editable via manual editing, file name for settings: server_settings.json.",
                        ),
                    )
                    new_address_input = gr.Textbox(
                        label=translations.get("new_server_address", "New Server Address"),
                        placeholder=translations.get("new_server_address_msg", "Ex. http://127.0.0.1"),
                    )
                    new_port_input = gr.Number(label=translations.get("new_server_port", "Port"), value=11434)
                    new_default_server = gr.Checkbox(
                        label=translations.get("default_start_setting", "Make the new host by default."),
                        value=False,
                    )
                    add_server_button = gr.Button(value=translations.get("add_ai_server", "Add Ai Server"))

                with gr.Accordion("Advanced", open=False):
                    language_dropdown = gr.Dropdown(
                        choices=list(language_settings["languages"].keys()),
                        label="Language",
                        value=current_language,
                        interactive=True,
                    )
                    llm_temperature_view = gr.Slider(
                        0,
                        1,
                        step=0.1,
                        value=0.5,
                        label="Temperature",
                        info="Choose between 0 and 1",
                        interactive=True,
                    )
                    llm_max_tokens_view = gr.Number(label="Max Tokens", interactive=True, value=1500)
                    prompt_temp_area = gr.Textbox(
                        label="Prompt workspace",
                        show_label=False,
                        lines=8,
                        buttons=["copy"],
                        interactive=True,
                        placeholder=translations.get("prompt_temp_area", "Temporary workspace for the prompt."),
                    )
                    with gr.Row():
                        clean_button = gr.ClearButton(prompt_temp_area, value=translations.get("clean_button", "Clean"))
                        prompt_button = gr.Button(value=translations.get("prompt_button", "Send prompt"))
                    gr.Markdown(
                        "**Tool Commands**  \n"
                        "`/tools`  \n"
                        "`/tool calculator {\"expression\":\"2+3*4\"}`  \n"
                        "`/tool datetime {}`  \n"
                        "`/tool web_search {\"query\":\"latest ollama news\",\"num_results\":3}`  \n"
                        "`/tool fetch_url {\"url\":\"https://example.com\"}`"
                    )

                server_name = gr.Textbox(
                    label=translations.get("server_address", "Server Address"),
                    value=current_host,
                    interactive=False,
                    visible=False,
                )

        # Sync drawer controls to hidden runtime inputs.
        search_provider_dropdown.change(lambda x: x, inputs=[search_provider_dropdown], outputs=[search_provider_state], queue=False)
        search_num_results_slider.change(lambda x: int(x), inputs=[search_num_results_slider], outputs=[search_num_results_state], queue=False)
        llm_temperature_view.change(lambda x: x, inputs=[llm_temperature_view], outputs=[llm_temperature], queue=False)
        llm_max_tokens_view.change(lambda x: x, inputs=[llm_max_tokens_view], outputs=[llm_max_tokens], queue=False)

        web_search_toggle.click(
            fn=toggle_web_search,
            inputs=[web_search_enabled],
            outputs=[web_search_enabled, web_search_toggle],
            queue=False,
        )

        settings_toggle_button.click(
            fn=toggle_settings_drawer,
            inputs=[settings_open_state],
            outputs=[settings_open_state, settings_drawer, settings_toggle_button],
            queue=False,
        )

        save_search_setting_button.click(
            fn=save_search_settings_ui,
            inputs=[search_provider_dropdown, tavily_api_key_input, serper_api_key_input, search_num_results_slider],
            outputs=[status_output],
            queue=False,
        )

        add_server_button.click(
            handle_add_server,
            inputs=[new_server_name_input, new_address_input, new_port_input, new_default_server],
            outputs=[status_output, server_dropdown, model_dropdown, new_server_name_input, new_address_input],
        )

        server_dropdown.change(
            handle_server_change,
            inputs=[server_dropdown],
            outputs=[status_output, model_dropdown, server_name],
        )

        stop_button.click(
            fn=stop_response,
            inputs=[],
            outputs=[question_input],
            queue=False,
        )

        prompt_button.click(
            fn=send_prompt,
            inputs=[prompt_temp_area],
            outputs=[question_input],
            queue=False,
        )

        def on_language_change(selected_language):
            new_translations = update_language(selected_language)
            return (
                gr.update(label=new_translations.get("select_server", "Select Server")),
                gr.update(label=new_translations.get("server_address", "Server Address")),
                gr.update(label=new_translations.get("select_model", "Select Model")),
                gr.update(
                    label=new_translations.get("status_display", "Status Display"),
                    placeholder=new_translations.get(
                        "status_display_msg",
                        "Displaying the model's status or displaying an error message.",
                    ),
                ),
                gr.update(
                    label=new_translations.get("enter_question", "Enter Your Question"),
                    placeholder=new_translations.get(
                        "enter_question",
                        "Enter your question or upload image file with question...",
                    ),
                ),
                gr.update(
                    label=new_translations.get("ai_server_name", "Ai Server Name"),
                    placeholder=new_translations.get(
                        "ai_server_name_msg",
                        "Ex: GMK-K9, editable via manual editing, file name for settings: server_settings.json.",
                    ),
                ),
                gr.update(
                    label=new_translations.get("new_server_address", "New Server Address"),
                    placeholder=new_translations.get("new_server_address_msg", "Ex. http://127.0.0.1"),
                ),
                gr.update(label=new_translations.get("new_server_port", "Port")),
                gr.update(label=new_translations.get("default_start_setting", "Make the new host by default.")),
                gr.update(value=new_translations.get("add_ai_server", "Add Ai Server")),
                gr.update(value=new_translations.get("clean_button", "Clean")),
                gr.update(value=new_translations.get("stop_button", "Stop")),
                gr.update(value=new_translations.get("clean_answer_button", "Clear Answer")),
                gr.update(value=new_translations.get("prompt_button", "Send Prompt")),
                gr.update(placeholder=new_translations.get("prompt_temp_area", "Temporary workspace for the prompt.")),
                gr.update(value=selected_language),
            )

        language_dropdown.change(
            on_language_change,
            [language_dropdown],
            [
                server_dropdown,
                server_name,
                model_dropdown,
                status_output,
                question_input,
                new_server_name_input,
                new_address_input,
                new_port_input,
                new_default_server,
                add_server_button,
                clean_button,
                stop_button,
                clean_answer_button,
                prompt_button,
                prompt_temp_area,
                language_dropdown,
            ],
        )

    return demo
