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

- `GET /` dashboard
- `GET /test` interactive test page
- `POST /classify` classify one ticket
- `POST /classify-batch` classify multiple tickets
- `GET /audit-logs` view recent decisions
- `GET /low-confidence-tickets` manual-review queue helper
- `POST /reload-config` reload YAML rules without restart
- `GET /verify-ollama-connection` check model connectivity

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.lock
```

`requirements.lock` pins the complete runtime dependency graph used by CI.
Update it deliberately with `uv pip compile requirements.txt --python-version
3.11 --universal --output-file requirements.lock` when changing the direct
requirements.

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
- Audit-log responses are newest-first; malformed or oversized JSONL records are ignored so one damaged record cannot hide later decisions.
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
