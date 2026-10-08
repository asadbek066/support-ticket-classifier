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
- `POST /reload-config` re-read and validate `config.yaml` (rules, model, URL, limits) without restart
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
- audit, rate-limit and provider-queue bounds

Changes take effect on `POST /reload-config` (or restart).

## Audit log storage

Every classification appends one JSON line to
`audit_logs/classifications-YYYY-MM-DD.jsonl` (UTC date). Each record holds the
timestamp, the classification result, and the ticket fields. Defaults, all
under the `audit:` section of `config.yaml` (applied at startup and on
`/reload-config`):

| Setting | Default | Meaning |
| --- | --- | --- |
| `max_text_chars` | 300 | Stored characters per ticket text field; the rest is discarded. |
| `store_full_text` | false | `true` stores complete ticket text (up to 20,000 characters per ticket, in plaintext). |
| `retention_days` | 30 | Files whose date is older are deleted. |
| `max_total_mb` | 200 | Total size of audit files; oldest files are deleted first. |
| `max_daily_mb` | 20 | Per-day file cap; once reached, further records that day are dropped. |
| `prune_interval_seconds` | 3600 | Retention runs at startup and at most this often during writes. |

Stored per ticket by default: the first 300 characters of each text field
(`subject`, `description`, `source_channel`, `customer_type`, `language`),
`ticket_digest` (SHA-256 and original character count of each full field, so a
record can be matched to the original ticket), and `truncated_fields`. The
digest of a short or empty field is also stored. Records written before this
change contain full text and stay readable; they age out with retention.

Retention only touches regular files named `classifications-YYYY-MM-DD.jsonl`
directly inside the audit directory (symlinks and other files are ignored) and
never deletes the file being written. When the daily cap is hit the request is
still classified; the record is dropped, one warning is logged per process per
day, and `GET /audit-logs` reports `dropped_today` and `dropped_total`
(counted per process since start). Invalid `audit:` values make `/reload-config`
fail rather than silently weaken a bound.

## Abuse controls for the classify endpoints

`/classify` and `/classify-batch` do not require authentication by default;
keep them on a trusted network (see below). Three bounds are available:

- Per-client rate limit (on by default, `rate_limit:` in `config.yaml`): an
  in-process token bucket keyed on the client address the application sees.
  Defaults: 300 tickets per minute sustained, burst 200. A batch is charged one
  token per ticket (so `burst` must be at least 100). Excess requests get `429`
  with a `Retry-After` header. Limits are per process: with several workers each
  keeps its own buckets. Behind a reverse proxy the address the app sees is the
  proxy's, so all users share one bucket unless forwarded headers are
  configured at the server (for example uvicorn `--proxy-headers` with
  `--forwarded-allow-ips` set to your proxy only). The app itself never reads
  `X-Forwarded-For`.
- Provider queue (`provider_queue:`), described in the Notes below.
- Optional shared secret: set `TICKET_CLASSIFIER_CLASSIFY_TOKEN` (16 to 256
  ASCII characters) and callers must send it in `X-Classify-Token` (constant
  time comparison, `401` otherwise; a set but too-short value answers `503`).
  Unset or empty leaves the endpoints open. Recommended whenever the service
  is reachable by anything other than the trusted caller.

## Notes

- Classification quality depends on prompt/model quality and configured rules.
- Ticket fields are treated as untrusted prompt data. Explicit attempts to change classification instructions go to `Other / Needs Review`; text with common security-incident indicators is routed to `Security Concerns` even if the model selects another category. These deterministic checks are conservative safety nets, not complete detectors.
- `POST /classify` is bounded to 90 seconds and each batch item to 60 seconds
  inside the 120-second batch budget; timed-out calls are reported and audited
  as `Other / Needs Review` rather than failing the request. Audit writes are
  separately bounded at 5 seconds each, so a stalled audit filesystem can add
  up to 5 seconds per item beyond the provider budget. Because each
  classification runs a real model call, keep batches small enough for the
  model's throughput (the 100-ticket schema maximum is a hard cap, not a
  latency promise).
- At most four Ollama calls run concurrently per process. Timed-out requests
  close their HTTP client. When all four slots are busy, further requests wait
  in a bounded first-in-first-out queue (`provider_queue.max_depth`, default 32;
  `provider_queue.max_wait_seconds`, default 30). Only when the queue is full or
  the wait expires does the call use the human-review fallback, and that
  fallback is marked: its `reason` is (or starts with) `provider_busy` in the
  response and the audit record, and batch results count it in `degraded`.
  The HTTP status stays 200, as for other fallbacks. A request cancelled while
  queued gives up its place.
  Loopback, private-use `.internal`, and private-IP Ollama calls bypass
  environment proxies so ticket text stays on the local route. Configured
  hostnames otherwise follow environment proxy settings; add custom private DNS
  names to `NO_PROXY` when they must be reached directly.
- `/classify-batch` returns a `degraded` count alongside `results`; degraded
  entries are persisted to the audit log so dashboard totals match the
  responses the caller received.
- `/verify-ollama-connection` probes `GET /api/tags` and reports
  `model_available`; it never runs an inference. Use `/health` for plain
  liveness.
- Audit-log responses are newest-first; malformed or oversized JSONL records
  are ignored so one damaged record cannot hide later decisions. Audit writes
  are best effort and limited to two concurrent writes on a separate executor:
  a failed or saturated write is logged and classification still returns, so
  monitor the service log if audit durability matters. On POSIX systems, the
  audit directory and files are restricted to the service account (`0700` and
  `0600`). Counts are maintained incrementally per file and reads scan from the
  newest entries. On Windows, restrict the `audit_logs` directory ACL to the
  service account and protect its parent directory against replacement by
  other accounts before running with real ticket data; the service does not
  rewrite NTFS ACLs.
- Configuration is loaded from `config.yaml` at startup and again only when
  `POST /reload-config` is called; editing the file does nothing by itself.
  A reload parses and validates the whole file (categories, queue map, rules,
  Ollama model and URL, and the `audit`, `rate_limit` and `provider_queue`
  sections) and then switches every setting over together. If the file is
  missing, malformed, or any section is invalid, the reload answers `503` and
  the previous model, URL, rules and limits stay active. A request uses one
  configuration generation from start to finish, including across a concurrent
  reload. A service whose config is invalid at startup refuses to start.
- Dependabot tracks `requirements.txt` only; it cannot regenerate the
  hash-pinned uv lock, so recompile the lock when accepting a dependency bump.
- Request bodies are capped at 8 MB, and responses carry `no-store` plus
  `nosniff`/`DENY`/CSP headers.
- Audit retention is built in (see "Audit log storage"); keep configuration changes under version control and apply any stricter organisational retention outside the service.

## Security and operating boundary

Admin audit, manual-review, config-reload, and Ollama-diagnostic endpoints
require the configured `TICKET_CLASSIFIER_ADMIN_TOKEN` in the
`X-Admin-Token` header and fail closed when it is absent. Keep the service on
the loopback interface (`127.0.0.1`, the default Uvicorn bind) or put it behind
an authenticated, trusted reverse proxy; do not expose the dashboard or
classification endpoints directly to the internet without an abuse-control
and privacy review. Ticket text and model output are treated as untrusted data,
and the API bounds ticket fields and batch size before processing.
