# Architecture Snapshot

This document reflects the current runtime split after the staged refactor away from a monolithic `chat_service.py`.

## Current Structure

```text
app/
  main.py
  core/
    app_settings.py
    cancellation.py
    config.py
    logging_setup.py
    storage.py
    tool_router.py
  orchestrator/
    __init__.py
    auto_tool_planner.py
    context.py
    conversation_pipeline.py
    intent_router.py
    model_runtime.py
    orchestrator.py
    policy.py
    response_renderer.py
    tool_runtime.py
    types.py
  services/
    chat_service.py
    persona_service.py
    preset_service.py
    prompt_service.py
    server_service.py
    session_service.py
  tools/
    base.py
    registry.py
    implementations/
    search_providers/
  ui/
    gradio_app.py
data/
  sessions.json
  presets.json
  personas.json
  prompts.json
exports/
```

## Runtime Responsibilities

- `app/main.py`
  - Process entry point. Configures logging and launches the Gradio app.
- `app/ui/gradio_app.py`
  - Builds the UI, localized labels, and event wiring.
- `app/services/chat_service.py`
  - UI-facing adapter for chat flow, streaming output, tool command handling, and markdown export.
- `app/services/server_service.py`
  - Ollama host management, model loading, and connection tests.
- `app/services/session_service.py`
  - Session CRUD backed by `data/sessions.json`.
- `app/services/preset_service.py`
  - LLM preset CRUD backed by `data/presets.json`.
- `app/services/persona_service.py`
  - Persona CRUD backed by `data/personas.json`.
- `app/services/prompt_service.py`
  - Prompt library CRUD backed by `data/prompts.json`.

## Orchestrator Responsibilities

- `intent_router.py`
  - Deterministic detection for time, calculator, fetch, and search intents.
- `auto_tool_planner.py`
  - Converts detected intent into direct tool execution or model-assisted follow-up.
- `tool_runtime.py`
  - Executes tools behind policy and cancellation checks.
- `model_runtime.py`
  - Builds summarization prompts and handles model-call runtime concerns.
- `conversation_pipeline.py`
  - Holds legacy multi-step conversation flow used during the migration.
- `orchestrator.py` and `types.py`
  - Shared typed orchestration flow and outputs.

## Persistence Model

- Root config files:
  - `server_settings.json`
  - `app_settings.json`
  - `language_settings.json`
- JSON data store under `data/`:
  - sessions
  - presets
  - personas
  - prompts
- Exported conversations are written to `exports/`.

## Current Status

- The refactor is functionally in place.
- `chat_service.py` is still the main integration layer for the UI and retains some legacy flow coordination.
- The orchestrator and service split already defines the practical architecture boundary for ongoing work.
