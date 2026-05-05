# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gradio-based WebUI for Ollama, supporting multi-host management, multimodal chat, deterministic tool routing, web search (Serper.dev / Tavily), and i18n (Chinese / English / Thai). The UI is Chinese-first with full English and Thai translations.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python ollama-webui.py

# Syntax check all source
python -m compileall app

# Run all tests
python -m unittest discover -v tests

# Run a single test module
python -m unittest -v tests/test_smoke_workflows.py
python -m unittest -v tests/test_phase1_services.py
python -m unittest -v tests/test_phase2_chat_sessions.py
```

No linter or formatter is configured. Only two dependencies: `gradio==6.9.0` and `Requests==2.32.5`.

## Architecture

### Request Flow

User input → `chat_service.ask_question_stream()` → the following decision chain:

1. **Manual tool commands** (`/tools`, `/tool <name> <json-args>`) — handled inline, return immediately
2. **Deterministic tool routing** via `AutoToolPlanner.plan()`:
   - `ToolIntentRouter.route()` checks for time/calc/fetch/search patterns (regex + keyword matching in Chinese and English)
   - `ToolRuntime.execute()` runs the matched tool through `ToolPolicy` then `ToolRegistry`
   - `DirectAction` results (datetime, calculator) are returned directly to user
   - `FetchForLlmAction` / `SearchForLlmAction` results go to step 3
3. **Model-based summarization** — fetched URLs and search results are summarized via a second Ollama call with a tailored prompt (`build_fetch_summary_prompt` / `build_search_summary_prompt` in `model_runtime.py`)
4. **Legacy conversation pipeline** (`conversation_pipeline.py`) — for non-tool questions: sends the prompt with tool instructions to the model, then parses the model's response for `<tool_call>...考上...</think>` tool-call markup. If a tool call is found, executes it and sends the result back for a natural-language explanation. If not, renders the model's direct answer. Falls back through multiple payload variants if the model fails.

### Layered Orchestration (migration in progress)

The codebase is mid-refactor from a monolithic `chat_service.py` to a layered orchestrator:

- `app/orchestrator/` — deterministic routing layer (scaffolded, partially wired)
  - `intent_router.py` — rule-based intent detection
  - `auto_tool_planner.py` — bridges router output to typed actions
  - `tool_runtime.py` — policy-gated tool execution
  - `policy.py` — allowed-tools whitelist and trusted domains
  - `context.py` — `RequestContext`, `ModelOptions`, `SearchSettings` dataclasses (not yet wired into main flow)
  - `orchestrator.py` — `Orchestrator.process()` entry point (scaffolded, not yet the primary path)
  - `types.py` — `ToolCall`, `ToolExecutionResult`, `OrchestratorOutput`, tagged union `AutoToolAction`
- `chat_service.py` — still the primary adapter driving the Gradio streaming interface, calling orchestrator components directly

The two paths coexist: deterministic intents go through `AutoToolPlanner`, while the legacy "ask model for tool decision" flow lives in `conversation_pipeline.py`. See `docs/ARCHITECTURE_REFACTOR.md` for the migration plan.

### Key Modules

| Path | Role |
|------|------|
| `ollama-webui.py` | Entry point → `app.main.launch()` |
| `app/main.py` | Logging + Gradio launch |
| `app/ui/gradio_app.py` | Full UI layout, event wiring, i18n translations dict (~460 lines of translations) |
| `app/services/chat_service.py` | Streaming chat adapter, session/preset/persona management, Gradio callback hub |
| `app/services/server_service.py` | Model listing (`/v1/models`), add-server, server-change, connection test |
| `app/services/session_service.py` | JSON-backed session CRUD + message persistence |
| `app/services/preset_service.py` | LLM parameter presets (built-in + custom) |
| `app/services/persona_service.py` | Persona management (system prompt, default model/preset) |
| `app/services/prompt_service.py` | Prompt library (CRUD, categories, favorites) |
| `app/orchestrator/model_runtime.py` | Ollama HTTP calls (streaming + non-streaming), summary prompt builders |
| `app/orchestrator/conversation_pipeline.py` | Legacy two-pass tool-decision flow |
| `app/tools/registry.py` | `ToolRegistry` — name→tool lookup, `execute()` |
| `app/tools/base.py` | `BaseTool` ABC interface |
| `app/tools/implementations/` | `datetime_tool.py`, `calculator.py`, `fetch_url.py`, `web_search.py` |
| `app/tools/search_providers/` | `tavily_provider.py`, `serper_provider.py` |
| `app/core/config.py` | `server_settings.json` I/O, LLM parameter load/save with clamping |
| `app/core/app_settings.py` | `app_settings.json` I/O (search config, API keys) |
| `app/core/storage.py` | `JsonStore` — generic JSON file read/write in `data/` dir |
| `app/core/cancellation.py` | Thread-safe stop flag (`request_stop`/`clear_stop`/`ensure_not_stopped`) |

## Configuration Files

| File | Contents | Synced to UI |
|------|----------|-------------|
| `server_settings.json` | Host list, `llm_parameters` array | Yes (Server dropdown, Advanced → Save LLM Settings) |
| `app_settings.json` | `search.provider`, `search.num_results`, `search.summary_length`, API keys | Yes (Search settings autosave) |
| `language_settings.json` | UI translations per language, `default_language` | Yes (Language dropdown) |
| `data/sessions.json` | Chat sessions + messages | Yes |
| `data/presets.json` | LLM parameter presets | Yes |
| `data/personas.json` | Personas with system prompts | Yes |
| `data/prompts.json` | Prompt library entries | Yes |

`data/` and `app_settings.json` are gitignored. Tests use `tempfile.TemporaryDirectory` with `JsonStore` for isolation.

## Testing Patterns

- `unittest` with `unittest.mock.patch` — no pytest
- Services are injected into `chat_service` module-level singletons, so tests `patch.object(chat_service, "session_service", ...)` to swap stores
- `FakeRuntime` class in smoke tests simulates `ToolRuntime.execute()` with a dict of responses
- Search/web tools are mocked at the registry level; no live API calls in tests

## Important Conventions

- UI strings are Chinese-first; status messages like `"連線正常"`, `"連線失敗"`, `"已停止回答"` are user-facing
- Ollama API: `/api/chat` (streaming + non-stream), `/v1/models` for model listing
- LLM parameter keys in `server_settings.json`: `llm_temperature`, `llm_max_tokens`, `llm_top_p`, `llm_typical_p`, `llm_num_ctx`; mapped to Ollama's `temperature`, `num_predict`, `top_p`, `typical_p`, `num_ctx`
- Tool call markup in model responses: `<tool_call>{"name":"...","arguments":{...}}考上` (parsed by `ToolRouter` regex)
- The `Clear Answer` button creates a new chat session rather than clearing the current one
- Search auto-saves on dropdown/slider change; API key inputs save on blur
- Gradio version is pinned to 6.9.0 — API signature changes across major versions
