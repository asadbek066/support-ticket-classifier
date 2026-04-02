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
pip install -r requirements.txt
```

3. Start Ollama and ensure your model is available.

4. Run API:

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
- For production, keep audit logs and configuration changes under version control.
