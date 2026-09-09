import asyncio
import hmac
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit

import httpx
import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse

from app.audit import (
    count_audit_logs,
    get_low_confidence_tickets,
    log_classification,
    read_audit_logs,
)
from app.ollama_client import generate_classification
from app.rules import RulesEngine
from app.schemas import (
    BatchClassificationRequest,
    BatchClassificationResponse,
    ClassificationResponse,
    Ticket,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
LOGGER = logging.getLogger(__name__)
ADMIN_TOKEN_ENV = "TICKET_CLASSIFIER_ADMIN_TOKEN"  # nosec B105 - variable name only
MAX_ADMIN_TOKEN_CHARS = 256
BATCH_TIMEOUT_SECONDS = 120.0

app = FastAPI(title="Ticket Classifier")
rules = RulesEngine(str(CONFIG_PATH))


def load_config() -> dict:
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except (OSError, yaml.YAMLError) as exc:
        LOGGER.error("config_load_failed (%s)", type(exc).__name__)
        raise RuntimeError("configuration unavailable") from exc
    if not isinstance(config, dict):
        raise TypeError("configuration unavailable")
    return config


def runtime_config(config: Mapping[str, object]) -> tuple[list[str], str, str]:
    raw_categories = config.get("categories", [])
    categories = (
        [item for item in raw_categories if isinstance(item, str)]
        if isinstance(raw_categories, list)
        else []
    )
    raw_ollama = config.get("ollama", {})
    ollama = raw_ollama if isinstance(raw_ollama, dict) else {}
    model = ollama.get("model", "deepseek-r1:1.5b")
    api_url = ollama.get("api_url", "http://localhost:11434/api/generate")
    if not isinstance(model, str) or not model.strip() or len(model) > 200:
        raise RuntimeError("configuration unavailable")
    if not isinstance(api_url, str) or not api_url.strip() or len(api_url) > 500:
        raise RuntimeError("configuration unavailable")
    return categories, model.strip(), api_url.strip()


def safe_api_url(api_url: str) -> str:
    try:
        parsed = urlsplit(api_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return "configured endpoint"
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        netloc = host
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        return f"{parsed.scheme}://{netloc}{parsed.path}"
    except ValueError:
        return "configured endpoint"


def log_failure(label: str, exc: Exception) -> None:
    LOGGER.error("%s (%s)", label, type(exc).__name__)


def require_admin_token(
    admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> None:
    """Require an explicitly configured high-entropy token for admin surfaces."""

    expected = os.getenv(ADMIN_TOKEN_ENV)
    if (
        expected is None
        or not 16 <= len(expected) <= MAX_ADMIN_TOKEN_CHARS
        or any(
            ord(character) < 0x20 or ord(character) == 0x7F for character in expected
        )
    ):
        raise HTTPException(status_code=503, detail="admin access is not configured")
    if (
        admin_token is None
        or len(admin_token) > MAX_ADMIN_TOKEN_CHARS
        or not hmac.compare_digest(admin_token, expected)
    ):
        raise HTTPException(
            status_code=401,
            detail="admin authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _unavailable_classification() -> ClassificationResponse:
    return ClassificationResponse(
        category="Other / Needs Review",
        confidence=0.0,
        queue="triage",
        reason="classification unavailable",
        human_review=True,
    )


@app.get("/")
def admin_dashboard():
    admin_file = ROOT / "app" / "admin.html"
    if admin_file.exists():
        return FileResponse(admin_file, media_type="text/html")
    return {"message": "Admin dashboard - visit http://localhost:8000/ to view"}


@app.get("/test")
def test_page():
    test_file = ROOT / "app" / "test.html"
    if test_file.exists():
        return FileResponse(test_file, media_type="text/html")
    return {"message": "Test page not found"}


@app.post("/classify", response_model=ClassificationResponse)
async def classify(ticket: Ticket):
    try:
        categories, model, api_url = runtime_config(load_config())
        ticket_data = ticket.model_dump()
        model_out = await asyncio.to_thread(
            generate_classification, ticket_data, categories, model, api_url
        )
        result = rules.apply(ticket_data, model_out)
        log_classification(ticket_data, result)
        return ClassificationResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        log_failure("classification_failed", exc)
        raise HTTPException(
            status_code=503, detail="classification unavailable"
        ) from exc


@app.post("/reload-config", dependencies=[Depends(require_admin_token)])
def reload_config():
    try:
        rules.reload()
        return {"status": "ok"}
    except Exception as exc:
        log_failure("config_reload_failed", exc)
        raise HTTPException(
            status_code=500, detail="configuration unavailable"
        ) from exc


@app.get("/verify-ollama-connection", dependencies=[Depends(require_admin_token)])
def verify_ollama():
    try:
        _, model, api_url = runtime_config(load_config())
    except Exception as exc:
        log_failure("config_load_failed", exc)
        raise HTTPException(
            status_code=503, detail="configuration unavailable"
        ) from exc

    payload = {"model": model, "prompt": "test", "stream": False}
    try:
        r = httpx.post(api_url, json=payload, timeout=10.0)
        if r.status_code == 200:
            return {
                "connected": True,
                "model": model,
                "api_url": safe_api_url(api_url),
                "status": "OK",
            }
        return {
            "connected": False,
            "status_code": r.status_code,
            "api_url": safe_api_url(api_url),
            "error": "Ollama returned an error",
        }
    except (httpx.HTTPError, OSError) as exc:
        log_failure("ollama_check_failed", exc)
        return {
            "connected": False,
            "api_url": safe_api_url(api_url),
            "model": model,
            "error": "Ollama is unavailable",
            "hint": "Is Ollama running? Check: (1) Start Ollama desktop app, (2) Or run 'ollama serve' in a terminal, (3) Check if api_url in config.yaml is correct.",
        }


@app.post("/classify-batch", response_model=BatchClassificationResponse)
async def classify_batch(req: BatchClassificationRequest):
    start = monotonic()
    results: list[ClassificationResponse] = []
    try:
        categories, model, api_url = runtime_config(load_config())
    except Exception as exc:
        log_failure("batch_config_load_failed", exc)
        raise HTTPException(
            status_code=503, detail="configuration unavailable"
        ) from exc

    try:
        async with asyncio.timeout(BATCH_TIMEOUT_SECONDS):
            for ticket in req.tickets:
                ticket_data = ticket.model_dump()
                try:
                    model_out = await asyncio.to_thread(
                        generate_classification, ticket_data, categories, model, api_url
                    )
                    result = rules.apply(ticket_data, model_out)
                    results.append(ClassificationResponse(**result))
                    log_classification(ticket_data, result)
                except Exception as exc:  # noqa: BLE001
                    # One malformed/provider-failed item must not discard the rest of a batch.
                    log_failure("batch_classification_item_failed", exc)
                    results.append(_unavailable_classification())
    except TimeoutError as exc:
        log_failure("batch_classification_timeout", exc)
        results.extend(
            _unavailable_classification() for _ in req.tickets[len(results) :]
        )

    elapsed_ms = (monotonic() - start) * 1000
    return BatchClassificationResponse(
        results=results, total=len(results), processing_time_ms=elapsed_ms
    )


@app.get("/audit-logs", dependencies=[Depends(require_admin_token)])
def get_audit_logs(limit: int = Query(default=100, ge=1, le=1000)):
    try:
        logs = read_audit_logs(limit=limit)
        return {"total": count_audit_logs(), "logs": logs}
    except Exception as exc:
        log_failure("audit_read_failed", exc)
        raise HTTPException(status_code=500, detail="audit logs unavailable") from exc


@app.get("/low-confidence-tickets", dependencies=[Depends(require_admin_token)])
def get_low_confidence(
    threshold: float = Query(default=0.65, ge=0.0, le=1.0),
    limit: int = Query(default=50, ge=1, le=1000),
):
    try:
        tickets = get_low_confidence_tickets(threshold=threshold, limit=limit)
        return {"threshold": threshold, "count": len(tickets), "tickets": tickets}
    except Exception as exc:
        log_failure("low_confidence_read_failed", exc)
        raise HTTPException(status_code=500, detail="review queue unavailable") from exc
