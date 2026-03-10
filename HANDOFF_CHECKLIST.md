# HANDOFF CHECKLIST

- Copy repo + `server_settings.json` + `app_settings.json` + `language_settings.json`
- Install deps: `pip install -r requirements.txt`
- Sanity check: `python -m compileall app`
- Start app: `python ollama-webui.py`
- Verify host/model dropdown loads correctly
- Verify `Clear Answer` clears history
- Verify `Stop Answer` can stop active response
- Verify Search settings save (`num_results`, `summary_length`)
- Verify LLM settings save to `server_settings.json`
