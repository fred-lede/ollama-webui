# HANDOFF

This file helps you continue development on another computer with minimal setup drift.

## 1. Workspace Snapshot
- Entry point: `ollama-webui.py`
- Main UI: `app/ui/gradio_app.py`
- Chat/runtime logic: `app/services/chat_service.py`
- Server/model connection checks: `app/services/server_service.py`
- Legacy flow pipeline: `app/orchestrator/conversation_pipeline.py`
- Config IO:
  - `app/core/config.py` (`server_settings.json`, `llm_parameters`)
  - `app/core/app_settings.py` (`app_settings.json`)
- Tool implementations:
  - `app/tools/implementations/*.py`
  - `app/tools/search_providers/*.py`

## 2. Files To Copy Between Computers
- Source code folder (entire repo)
- Runtime config files:
  - `server_settings.json`
  - `app_settings.json`
  - `language_settings.json` (if customized)
- Optional local log (for troubleshooting only):
  - `log-webui.log`

## 3. Environment Setup
1. Install Python 3.10+.
2. Open repo root.
3. (Optional) Create venv.
4. Install dependencies:
   `pip install -r requirements.txt`
5. Start app:
   `python ollama-webui.py`

## 4. Post-Setup Validation Checklist
- App launches without traceback.
- Host dropdown can load models from your default host.
- Main toolbar `連線測試` can verify `/api/chat` for the selected server+model.
- Connection status light updates correctly (`未測試` / `連線正常` / `連線失敗`).
- `Clear Answer` clears both visible chat and hidden history.
- `Stop Answer` stops active response and shows stopping status.
- Search settings are persisted:
  - `Search Results Count` (up to 20)
  - `Search Summary Length` (short/medium/long)
- LLM settings are persisted to `server_settings.json`:
  - `llm_temperature`
  - `llm_max_tokens`
  - `llm_top_p`
  - `llm_typical_p`
  - `llm_num_ctx`

## 5. Operational Notes
- `app_settings.json` may contain API keys for search providers. Keep this file private.
- `server_settings.json` may include internal host addresses.
- Search output may fall back to raw result listing if remote model summarization returns empty.

## 6. Quick Troubleshooting
- If startup fails due to UI args mismatch:
  - Verify installed Gradio version from environment.
  - Reinstall with `pip install -r requirements.txt`.
- If search fails:
  - Check API keys in `app_settings.json`.
  - Confirm provider endpoint URLs.
- If remote Ollama is unstable:
  - Use main toolbar `連線測試` first (tests selected server+model via `/api/chat`).
  - Then check server availability and model health.
  - Review `log-webui.log` for request errors.

## 7. Recommended First Commands On New Machine
```bash
python --version
pip install -r requirements.txt
python -m compileall app
python ollama-webui.py
```
