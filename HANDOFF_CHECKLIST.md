# HANDOFF CHECKLIST

- Copy repo + `server_settings.json` + `app_settings.json` + `language_settings.json`
- Copy `data/` if you need sessions, presets, personas, and prompt library entries
- Install deps: `pip install -r requirements.txt`
- Sanity check: `python -m compileall app`
- Start app: `python ollama-webui.py`
- Alternative start: `python -m app.main`
- Verify host/model dropdown loads correctly
- Verify `連線測試` can check selected server+model via `/api/chat`
- Verify connection status light transitions (`未測試` / `連線正常` / `連線失敗`)
- Verify `Clear Answer` clears history
- Verify `Stop Answer` can stop active response
- Verify Search settings save (`num_results`, `summary_length`)
- Verify LLM settings save to `server_settings.json`
- Verify session/preset/persona/prompt data persists in `data/*.json`
