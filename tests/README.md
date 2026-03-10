# Tests

## Smoke Test Suite

Run:

```bash
python -m unittest -v tests/test_smoke_workflows.py
```

Covers:
- deterministic tool intents:
  - time
  - calculator
  - fetch_url
  - web_search
- stop behavior:
  - stop flag set
  - stale stop flag cleared on next request
  - stop during stream returns `Stopped`
- clear behavior:
  - clear history state helper output
