# Ollama WebUI (Gradio)

[Chinese](./README.md) | English

## Overview
This project provides a Gradio-based WebUI for local or remote Ollama servers, including chat, vision input, web search, and tool integration.

## Features
- Multi-language UI (Chinese, English, Thai; extendable via `language_settings.json`)
- Multi-host management (hosts loaded from `server_settings.json`; add host from UI)
- LLM parameter controls:
  - `Temperature`
  - `Max Tokens`
  - `Top P`
  - `Typical P`
  - `Num CTX`
- Web Search providers: `Serper.dev` and `Tavily`
- Search controls:
  - `Search Results Count` (1~20)
  - `Search Summary Length` (`short` / `medium` / `long`)
- Built-in tools:
  - `datetime`
  - `calculator`
  - `fetch_url`
  - `web_search`
- Manual tool commands:
  - `/tools`
  - `/tool <name> <json-args>`
- Main toolbar provides `連線測試` (Connection Test) with status light (`未測試` / `連線正常` / `連線失敗`)
- `Stop Answer` supports cancellation for streaming and most tool paths
- `Clear Answer` clears both visible chat and internal history

## Architecture (Orchestrator)
The codebase has been progressively split from a monolithic `chat_service.py` into layered components:
- `app/services/chat_service.py`
  - Gradio interaction + streaming adapter
- `app/orchestrator/auto_tool_planner.py`
  - deterministic tool planning + fallback strategy
- `app/orchestrator/intent_router.py`
  - intent detection (time/calc/fetch/search)
- `app/orchestrator/tool_runtime.py`
  - policy-guarded tool execution with cancellation support
- `app/orchestrator/model_runtime.py`
  - model request runtime + summary prompt builders
- `app/orchestrator/conversation_pipeline.py`
  - extracted legacy conversation flow (tool decision/fallback/tool-call/second model pass)
- `app/orchestrator/types.py`
  - typed actions and orchestration data structures

This separation improves maintainability, testability, and change safety across UI, tool, and model layers.

## Config Files
- `server_settings.json`
  - `hosts`: Ollama host list
  - `llm_parameters`: default LLM parameters (synced with UI)
- `app_settings.json`
  - `search.provider`
  - `search.num_results`
  - `search.summary_length`
  - `search.tavily_api_key` / `search.serper_api_key`
  - `search.tavily_api_url` / `search.serper_api_url`

## Requirements
- Python 3.10+ (recommended)
- Install dependencies from `requirements.txt`

## Install & Run
1. (Optional) Create a virtual environment
2. Install dependencies:
   `pip install -r requirements.txt`
3. Start:
   `python ollama-webui.py`

## Common Tasks
- Switch/add Ollama hosts: Settings drawer > `Server`
- Tune search & summary length: Settings drawer > `Search`
- Save LLM defaults back to file: Settings drawer > `Advanced` > `Save LLM Settings`
- Test active model connectivity: main toolbar > `連線測試` (status shown in light + status textbox)

## Additional Docs
- Chinese readme: [README.md](./README.md)
- Cross-machine handoff guide: [HANDOFF.md](./HANDOFF.md)

## License
This project is licensed under [MIT License](LICENSE).
