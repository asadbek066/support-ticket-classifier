# Support Ticket Classifier

Rule-assisted AI service for support ticket classification and queue routing.

## Overview

This API classifies incoming support tickets into configured categories,
maps them to internal queues, and flags low-confidence cases for manual review.
It is designed for local-first usage with Ollama.

## Features

- Single ticket classification endpoint
- Batch classification endpoint
- Queue mapping via `config.yaml`
- Confidence threshold and forced human-review rules
- Audit logging for classification decisions
- Lightweight dashboard and test page

## Tech Stack

- Python
- FastAPI
- Ollama (local LLM)
- YAML-based routing rules

## API Endpoints

- `GET /health` liveness probe (no admin token, no Ollama call)
- `GET /` dashboard
- `GET /test` interactive test page
- `POST /classify` classify one ticket
- `POST /classify-batch` classify multiple tickets
- `GET /audit-logs` view recent decisions
- `GET /low-confidence-tickets` manual-review queue helper
- `GET /low-confidence-count` manual-review queue size only
- `POST /reload-config` reload YAML rules without restart
- `GET /verify-ollama-connection` check server connectivity and model presence

## Setup

1. Create and activate a virtual environment with Python 3.11 or newer
   (CI exercises 3.11, 3.12, and 3.13).
2. Install dependencies:

```bash
# The lock is hash-pinned: pip verifies every artifact before installing.
python -m pip install --require-hashes -r requirements.lock
```

`requirements.lock` pins the complete runtime dependency graph used by CI.
Update it deliberately with the command recorded at its top when changing the
direct requirements:

```bash
uv pip compile requirements.txt --python-version 3.11 --universal \
  --generate-hashes --output-file requirements.lock
```

3. Start Ollama and ensure your model is available.

4. Configure a high-entropy admin token for audit, review-queue, config-reload,
   and Ollama-diagnostic endpoints:

```bash
export TICKET_CLASSIFIER_ADMIN_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

   Send it in the `X-Admin-Token` header. The admin endpoints remain disabled
   with a `503` response until this token is configured.

5. Run API:

```bash
uvicorn app.main:app --reload --port 8000
```

## Quick Test

PowerShell example:

```powershell
$body = @{
  subject = "Payment failed"
  description = "Card declined on checkout"
  source_channel = "email"
  customer_type = "paid"
  language = "en"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/classify" -Method Post -ContentType "application/json" -Body $body
```

## Configuration

`config.yaml` defines:

- category list
- queue mapping
- confidence threshold
- forced human-review categories
- Ollama model and API URL

## Notes

- Classification quality depends on prompt/model quality and configured rules.
- `POST /classify` is bounded to 90 seconds and each batch item to 60 seconds
  inside the 120-second batch budget; timed-out calls are reported and audited
  as `Other / Needs Review` rather than failing the request. Audit writes are
  separately bounded at 5 seconds each, so a stalled audit filesystem can add
  up to 5 seconds per item beyond the provider budget. Because each
  classification runs a real model call, keep batches small enough for the
  model's throughput (the 100-ticket schema maximum is a hard cap, not a
  latency promise).
- `/classify-batch` returns a `degraded` count alongside `results`; degraded
  entries are persisted to the audit log so dashboard totals match the
  responses the caller received.
- `/verify-ollama-connection` probes `GET /api/tags` and reports
  `model_available`; it never runs an inference. Use `/health` for plain
  liveness.
- Audit-log responses are newest-first; malformed or oversized JSONL records
  are ignored so one damaged record cannot hide later decisions. Audit writes
  are best effort: a failed write is logged and the classification still
  returns, so monitor the service log if audit durability matters. Counts are
  maintained incrementally per file and reads scan from the newest entries.
- Configuration is cached by file mtime/size. A malformed or missing config
  keeps the last known good configuration active until `/reload-config`
  succeeds; prompt categories and rule validation always use the same rules
  snapshot, including across a concurrent reload.
- Dependabot tracks `requirements.txt` only; it cannot regenerate the
  hash-pinned uv lock, so recompile the lock when accepting a dependency bump.
- Request bodies are capped at 8 MB, and responses carry `no-store` plus
  `nosniff`/`DENY`/CSP headers.
- For production, keep audit logs and configuration changes under version control and configure retention outside the service.

## Security and operating boundary

Admin audit, manual-review, config-reload, and Ollama-diagnostic endpoints
require the configured `TICKET_CLASSIFIER_ADMIN_TOKEN` in the
`X-Admin-Token` header and fail closed when it is absent. Keep the service on
the loopback interface (`127.0.0.1`, the default Uvicorn bind) or put it behind
an authenticated, trusted reverse proxy; do not expose the dashboard or
classification endpoints directly to the internet without an abuse-control
and privacy review. Ticket text and model output are treated as untrusted data,
and the API bounds ticket fields and batch size before processing.
