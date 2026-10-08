# support-ticket-classifier

FastAPI service that classifies support tickets with a local Ollama model, then applies
deterministic rules from `config.yaml` to pick a category and queue, forcing human review on
low confidence and on some flagged text. Every decision is appended to `audit_logs/*.jsonl`.

## Commands

CI runs these on Python 3.11, 3.12 and 3.13:

```bash
python -m pip install --require-hashes -r requirements.lock
python -m unittest discover -s tests -v
python -m pip check
python -m py_compile app/*.py
```

No lint or typecheck step is configured. To run the API locally:

```bash
uvicorn app.main:app --reload --port 8000
```

## Layout

- `app/main.py`: FastAPI app, routes, admin-token check, config cache, audit executor.
- `app/rules.py`: `RulesEngine`; prompt-override and security-indicator checks, confidence adjustments, review rules.
- `app/ollama_client.py`: prompt building and HTTP calls to Ollama; caps concurrent calls at four.
- `app/audit.py`: JSONL audit writes and reads under `audit_logs/`, plus low-confidence queue helpers.
- `app/schemas.py`: request and response models, including the 100-ticket batch limit.
- `config.yaml`: categories, queue mapping, confidence threshold, forced-review categories, Ollama model and API URL.
- `tests/`: `unittest` suite that CI runs with `discover -s tests`.
- `requirements.txt` (direct dependencies) and `requirements.lock` (hash-pinned graph that CI installs).

## Rules for changes

- Tests never contact a live Ollama server. The suite runs with Ollama stubbed, so new tests must stub it too.
- Change dependencies in `requirements.txt`, then regenerate the lock with the command recorded in the lock header:

  ```bash
  uv pip compile requirements.txt --python-version 3.11 --universal --generate-hashes --output-file requirements.lock
  ```

- Code must run on Python 3.11, the lowest version in the CI matrix.
- Admin routes are gated on the `TICKET_CLASSIFIER_ADMIN_TOKEN` environment variable (`ADMIN_TOKEN_ENV` in `app/main.py`).
- CI syntax-checks only `app/*.py`; keep new modules in that directory so the check covers them.
