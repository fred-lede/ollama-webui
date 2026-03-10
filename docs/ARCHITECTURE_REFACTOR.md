# Architecture Refactor Plan

## Target Structure

```text
app/
  orchestrator/
    __init__.py
    context.py
    types.py
    policy.py
    intent_router.py
    tool_runtime.py
    response_renderer.py
    orchestrator.py
  services/
    chat_service.py              # thin adapter only (legacy compatibility)
  tools/
    ...
  ui/
    gradio_app.py
  core/
    cancellation.py
    config.py
    app_settings.py
```

## Intent

Move from a monolithic `chat_service.py` to a layered runtime:
- Deterministic routing first
- Policy-guarded tool execution
- Model summarization/response as a separate stage
- Shared request context and typed outputs

## First Batch (already scaffolded)

- `context.py`
  - `RequestContext`
  - `ModelOptions`
  - `SearchSettings`
- `types.py`
  - `ToolCall`
  - `ToolExecutionResult`
  - `OrchestratorOutput`
  - `OrchestratorStep`
- `policy.py`
  - `ToolPolicy` (enabled tools, trusted domains, retry/timeout defaults)
- `intent_router.py`
  - deterministic rule router for `datetime/calculator/fetch_url/web_search`
- `tool_runtime.py`
  - one entry for policy-checked tool execution
- `response_renderer.py`
  - tool/error render helpers
- `orchestrator.py`
  - central `Orchestrator.process()` flow

## Migration Steps

1. Create compatibility adapter in `chat_service.py`
- Keep public `ask_question_stream()` signature unchanged.
- Build a `RequestContext` from UI inputs.
- Invoke `Orchestrator` for deterministic tool intents.

2. Move deterministic logic from `chat_service.py` to `intent_router.py`
- time/calc/url/search detection
- expression/url extraction helpers

3. Move tool execution wrappers to `tool_runtime.py`
- current registry calls
- cancellation checks
- policy enforcement

4. Keep model-only logic in `chat_service.py` temporarily
- tool decision prompts
- summary prompts
- streaming response

5. Final split
- extract model-calling to a dedicated `model_runtime.py`
- keep `chat_service.py` as transport adapter only

## Acceptance Criteria

- Existing UI behavior unchanged
- All current tool flows still pass manual checks
- `ask_question_stream()` reduced to orchestration + streaming adapter
- No direct tool registry calls in UI layer

## Suggested Next PR

- Wire `Orchestrator` into `chat_service.py` for deterministic intents only.
- Keep fallback to legacy behavior for non-deterministic cases.
- Add 6 smoke tests:
  - time
  - calculator
  - fetch_url
  - web_search
  - stop during tool flow
  - clear history then ask new question
