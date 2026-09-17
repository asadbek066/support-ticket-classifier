import asyncio
import hmac
import logging
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit

import httpx
import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from starlette.datastructures import MutableHeaders
from starlette.middleware.body_limit import RequestBodyLimitMiddleware

from app.audit import (
    count_audit_logs,
    count_low_confidence_tickets,
    get_low_confidence_tickets,
    log_classification,
    read_audit_logs,
)
from app.ollama_client import (
    check_ollama_connection,
    generate_classification,
    is_provider_fallback,
    provider_fallback,
)
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
BATCH_ITEM_TIMEOUT_SECONDS = 60.0
CLASSIFY_TIMEOUT_SECONDS = 90.0
AUDIT_WRITE_TIMEOUT_SECONDS = 5.0
MAX_REQUEST_BODY_BYTES = 8 * 1024 * 1024
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'"
    ),
}
NO_STORE_PATHS = {"/", "/test"}
# FastAPI's generated docs load assets from a CDN, so they cannot run under
# the API's default-src 'self' policy.
CSP_EXEMPT_PATHS = {"/docs", "/redoc", "/openapi.json"}

app = FastAPI(title="Ticket Classifier")
rules = RulesEngine(str(CONFIG_PATH))

_CONFIG_LOCK = threading.Lock()
_CONFIG_CACHE: dict[str, object] = {}


def load_config() -> dict:
    """Return the parsed config, cached by file mtime/size.

    A config that cannot be read or parsed falls back to the last known good
    value when one exists, so a bad edit cannot take classification down while
    the operator fixes it. Only the very first load fails hard.
    """
    try:
        stat = CONFIG_PATH.stat()
    except OSError as exc:
        LOGGER.error("config_stat_failed (%s)", type(exc).__name__)
        with _CONFIG_LOCK:
            cached = _CONFIG_CACHE.get("value")
        if isinstance(cached, dict):
            LOGGER.warning("config_using_last_known_good")
            return cached
        raise RuntimeError("configuration unavailable") from exc

    cache_key = (stat.st_mtime_ns, stat.st_size)
    with _CONFIG_LOCK:
        cached = _CONFIG_CACHE.get("value")
        if _CONFIG_CACHE.get("key") == cache_key and isinstance(cached, dict):
            return cached

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if not isinstance(config, dict):
            raise TypeError("configuration must be a mapping")
    except (OSError, yaml.YAMLError, TypeError) as exc:
        LOGGER.error("config_load_failed (%s)", type(exc).__name__)
        with _CONFIG_LOCK:
            cached = _CONFIG_CACHE.get("value")
        if isinstance(cached, dict):
            LOGGER.warning("config_using_last_known_good")
            return cached
        raise RuntimeError("configuration unavailable") from exc

    with _CONFIG_LOCK:
        _CONFIG_CACHE["key"] = cache_key
        _CONFIG_CACHE["value"] = config
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
    try:
        parsed_url = httpx.URL(api_url.strip())
    except Exception as exc:
        raise RuntimeError("configuration unavailable") from exc
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.host:
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
        or not expected.isascii()
        or any(
            ord(character) < 0x20 or ord(character) == 0x7F for character in expected
        )
    ):
        raise HTTPException(status_code=503, detail="admin access is not configured")
    if (
        admin_token is None
        or len(admin_token) > MAX_ADMIN_TOKEN_CHARS
        or not admin_token.isascii()
        or not hmac.compare_digest(admin_token, expected)
    ):
        raise HTTPException(
            status_code=401,
            detail="admin authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )


class SecurityHeadersMiddleware:
    """Attach conservative response headers to every API response."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    if name == "Content-Security-Policy" and path in CSP_EXEMPT_PATHS:
                        continue
                    headers.setdefault(name, value)
                if path not in NO_STORE_PATHS:
                    headers.setdefault("Cache-Control", "no-store")
            await send(message)

        await self.app(scope, receive, send_with_headers)


def _unavailable_classification() -> ClassificationResponse:
    return ClassificationResponse(
        category="Other / Needs Review",
        confidence=0.0,
        queue="triage",
        reason="classification unavailable",
        human_review=True,
    )


async def _audit_classification(ticket_data: dict, result: dict) -> None:
    """Persist an audit entry without discarding a completed classification."""
    try:
        await asyncio.wait_for(
            asyncio.to_thread(log_classification, ticket_data, result),
            timeout=AUDIT_WRITE_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        log_failure("audit_write_timeout", TimeoutError("audit write timed out"))
    except Exception as exc:  # noqa: BLE001 - audit storage is best effort
        log_failure("audit_write_failed", exc)


def _provider_settings() -> tuple[str, str]:
    try:
        _, model, api_url = runtime_config(load_config())
        return model, api_url
    except Exception as exc:
        log_failure("config_load_failed", exc)
        raise HTTPException(
            status_code=503, detail="configuration unavailable"
        ) from exc


@app.get("/health")
def health():
    return {"status": "ok"}


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
    # Prompt categories and rule validation share one config snapshot so a
    # concurrent /reload-config cannot make the model return a category the
    # rules engine no longer accepts.
    config = rules.snapshot()
    categories = rules.configured_categories(config)
    model, api_url = _provider_settings()
    ticket_data = ticket.model_dump()
    try:
        model_out = await asyncio.wait_for(
            asyncio.to_thread(
                generate_classification, ticket_data, categories, model, api_url
            ),
            timeout=CLASSIFY_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        log_failure("classification_timeout", exc)
        model_out = provider_fallback()
    except Exception as exc:
        log_failure("classification_failed", exc)
        raise HTTPException(
            status_code=503, detail="classification unavailable"
        ) from exc
    result = rules.apply(ticket_data, model_out, config)
    await _audit_classification(ticket_data, result)
    return ClassificationResponse(**result)


@app.post("/reload-config", dependencies=[Depends(require_admin_token)])
def reload_config():
    try:
        rules.reload()
        return {"status": "ok"}
    except Exception as exc:
        log_failure("config_reload_failed", exc)
        raise HTTPException(
            status_code=503, detail="configuration unavailable"
        ) from exc


@app.get("/verify-ollama-connection", dependencies=[Depends(require_admin_token)])
def verify_ollama():
    model, api_url = _provider_settings()
    check = check_ollama_connection(api_url, model)
    return {
        "model": model,
        "api_url": safe_api_url(api_url),
        **check,
    }


@app.post("/classify-batch", response_model=BatchClassificationResponse)
async def classify_batch(req: BatchClassificationRequest):
    start = monotonic()
    deadline = start + BATCH_TIMEOUT_SECONDS
    results: list[ClassificationResponse] = []
    degraded = 0
    config = rules.snapshot()
    categories = rules.configured_categories(config)
    model, api_url = _provider_settings()

    for ticket in req.tickets:
        ticket_data = ticket.model_dump()
        remaining = deadline - monotonic()
        if remaining <= 0:
            result = _unavailable_classification().model_dump()
            degraded += 1
        else:
            try:
                model_out = await asyncio.wait_for(
                    asyncio.to_thread(
                        generate_classification,
                        ticket_data,
                        categories,
                        model,
                        api_url,
                    ),
                    timeout=min(BATCH_ITEM_TIMEOUT_SECONDS, remaining),
                )
                result = rules.apply(ticket_data, model_out, config)
                if is_provider_fallback(model_out):
                    degraded += 1
            except TimeoutError as exc:
                log_failure("batch_classification_item_timeout", exc)
                result = _unavailable_classification().model_dump()
                degraded += 1
            except Exception as exc:  # noqa: BLE001
                # One malformed/provider-failed item must not discard the rest.
                log_failure("batch_classification_item_failed", exc)
                result = _unavailable_classification().model_dump()
                degraded += 1
        # Degraded results are persisted too, so audit totals match the
        # results the caller received.
        await _audit_classification(ticket_data, result)
        results.append(ClassificationResponse(**result))

    elapsed_ms = (monotonic() - start) * 1000
    return BatchClassificationResponse(
        results=results,
        total=len(results),
        degraded=degraded,
        processing_time_ms=elapsed_ms,
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


@app.get("/low-confidence-count", dependencies=[Depends(require_admin_token)])
def get_low_confidence_count(
    threshold: float = Query(default=0.65, ge=0.0, le=1.0),
):
    try:
        count = count_low_confidence_tickets(threshold=threshold)
        return {"threshold": threshold, "count": count}
    except Exception as exc:
        log_failure("low_confidence_count_failed", exc)
        raise HTTPException(status_code=500, detail="review queue unavailable") from exc


# Body limits are added first so the security-header middleware wraps the
# early 413 response; both still run before request parsing.
app.add_middleware(RequestBodyLimitMiddleware, max_body_size=MAX_REQUEST_BODY_BYTES)
app.add_middleware(SecurityHeadersMiddleware)
