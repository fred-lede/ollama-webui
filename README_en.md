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
- `Stop Answer` supports cancellation for streaming and most tool paths
- `Clear Answer` clears both visible chat and internal history

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

## Additional Docs
- Chinese readme: [README.md](./README.md)
- Cross-machine handoff guide: [HANDOFF.md](./HANDOFF.md)

## License
This project is licensed under [MIT License](LICENSE).
