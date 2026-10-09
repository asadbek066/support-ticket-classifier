import asyncio
import hmac
import logging
import os
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit

import httpx
import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.middleware.body_limit import RequestBodyLimitMiddleware

from app.abuse import (
    ProviderQueueSettings,
    RateLimitSettings,
    TokenBucketLimiter,
    provider_queue_from_config,
    rate_limit_from_config,
)
from app.audit import (
    AuditSettings,
    count_audit_logs,
    count_low_confidence_tickets,
    drop_stats,
    get_low_confidence_tickets,
    log_classification,
    read_audit_logs,
    settings_from_config,
    startup_prune,
)
from app.audit import (
    configure as configure_audit,
)
from app.ollama_client import (
    BUSY_REASON,
    acquire_provider_call,
    check_ollama_connection,
    generate_classification,
    is_provider_fallback,
    provider_busy_fallback,
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
CLASSIFY_TOKEN_ENV = "TICKET_CLASSIFIER_CLASSIFY_TOKEN"  # nosec B105 - name only
MAX_ADMIN_TOKEN_CHARS = 256
BATCH_TIMEOUT_SECONDS = 120.0
BATCH_ITEM_TIMEOUT_SECONDS = 60.0
CLASSIFY_TIMEOUT_SECONDS = 90.0
AUDIT_WRITE_TIMEOUT_SECONDS = 5.0
MAX_CONCURRENT_AUDIT_WRITES = 2
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

rules = RulesEngine(str(CONFIG_PATH))

_LIMITER = TokenBucketLimiter(RateLimitSettings())
_QUEUE_SETTINGS = ProviderQueueSettings()
_ACTIVE: "ActiveConfig | None" = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if classify_token_misconfigured():
        LOGGER.error(
            "%s is set but empty, too short, or invalid; classify endpoints "
            "answer 503 until it is fixed or the variable is unset",
            CLASSIFY_TOKEN_ENV,
        )
    try:
        await asyncio.to_thread(startup_prune)
    except Exception as exc:  # noqa: BLE001 - housekeeping must not block startup
        log_failure("audit_startup_prune_failed", exc)
    yield


app = FastAPI(title="Ticket Classifier", lifespan=lifespan)

_CONFIG_LOCK = threading.Lock()
_AUDIT_EXECUTOR = ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_AUDIT_WRITES,
    thread_name_prefix="ticket-audit",
)
_AUDIT_ADMISSION = threading.BoundedSemaphore(MAX_CONCURRENT_AUDIT_WRITES)


@dataclass(frozen=True)
class ActiveConfig:
    """One validated generation of every config-derived setting."""

    raw: dict
    audit: AuditSettings
    rate_limit: RateLimitSettings
    provider_queue: ProviderQueueSettings


def read_config_file() -> dict:
    """Read and parse config.yaml; raise RuntimeError if it cannot be used."""
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if not isinstance(config, dict):
            raise TypeError("configuration must be a mapping")
    except (OSError, yaml.YAMLError, TypeError) as exc:
        LOGGER.error("config_load_failed (%s)", type(exc).__name__)
        raise RuntimeError("configuration unavailable") from exc
    return config


def build_active_config(raw: dict) -> ActiveConfig:
    """Validate every section of a parsed config as a whole."""
    try:
        runtime_config(raw)
        if not isinstance(raw.get("categories", []), list):
            raise ValueError("categories must be a list")
        for section in ("rules", "queue_map"):
            if not isinstance(raw.get(section, {}), dict):
                raise ValueError(f"{section} must be a mapping")
        return ActiveConfig(
            raw=raw,
            audit=settings_from_config(raw),
            rate_limit=rate_limit_from_config(raw),
            provider_queue=provider_queue_from_config(raw),
        )
    except ValueError as exc:
        LOGGER.error("config_invalid (%s)", exc)
        raise RuntimeError("configuration unavailable") from exc


def activate_config(active: ActiveConfig) -> None:
    """Install a validated generation: rules, provider, audit, limits together."""
    global _ACTIVE, _LIMITER, _QUEUE_SETTINGS
    with _CONFIG_LOCK:
        rules.cfg = active.raw
        configure_audit(active.audit)
        if _LIMITER.settings != active.rate_limit:
            _LIMITER = TokenBucketLimiter(active.rate_limit)
        _QUEUE_SETTINGS = active.provider_queue
        _ACTIVE = active


def reload_active_config() -> None:
    """Re-read config.yaml; on any failure the previous generation stays."""
    activate_config(build_active_config(read_config_file()))


def load_config() -> dict:
    """Return the active (last successfully loaded) config.

    The file is read only at startup and by /reload-config, so every setting
    comes from the same generation and a bad edit has no effect until a reload
    succeeds.
    """
    active = _ACTIVE
    if active is None:
        raise RuntimeError("configuration unavailable")
    return active.raw


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


# The first load fails hard: a service with no valid config must not start.
activate_config(build_active_config(rules.snapshot()))


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


def classify_token_misconfigured() -> bool:
    expected = os.getenv(CLASSIFY_TOKEN_ENV)
    return expected is not None and (
        not expected.strip()
        or not 16 <= len(expected) <= MAX_ADMIN_TOKEN_CHARS
        or not expected.isascii()
        or any(ord(c) < 0x20 or ord(c) == 0x7F for c in expected)
    )


def require_classify_token(
    classify_token: str | None = Header(default=None, alias="X-Classify-Token"),
) -> None:
    """Optional shared secret for the classify endpoints (off unless configured)."""

    expected = os.getenv(CLASSIFY_TOKEN_ENV)
    if expected is None:
        return
    if (
        not expected.strip()
        or not 16 <= len(expected) <= MAX_ADMIN_TOKEN_CHARS
        or not expected.isascii()
        or any(
            ord(character) < 0x20 or ord(character) == 0x7F for character in expected
        )
    ):
        raise HTTPException(
            status_code=503, detail="classification access is misconfigured"
        )
    if (
        classify_token is None
        or len(classify_token) > MAX_ADMIN_TOKEN_CHARS
        or not classify_token.isascii()
        or not hmac.compare_digest(classify_token, expected)
    ):
        raise HTTPException(status_code=401, detail="classification token required")


def _client_key(scope) -> str:
    client = scope.get("client")
    return client[0] if client else "unknown"


def _too_many(retry_after: int) -> JSONResponse:
    return JSONResponse(
        {"detail": "rate limit exceeded"},
        status_code=429,
        headers={"Retry-After": str(retry_after)},
    )


def rate_limit_batch_remainder(
    request: Request, req: BatchClassificationRequest
) -> None:
    """A valid batch pays for its other tickets; the first was charged up front."""
    extra = len(req.tickets) - 1
    if extra > 0:
        retry_after = _LIMITER.check(_client_key(request.scope), extra)
        if retry_after:
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )


class RateLimitMiddleware:
    """Charge one token per classify request before the body is parsed.

    Doing this ahead of validation means malformed or oversized bodies spend
    tokens like any other request on both endpoints.
    """

    PATHS = {"/classify", "/classify-batch"}

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and scope.get("path") in self.PATHS
        ):
            retry_after = _LIMITER.check(_client_key(scope), 1)
            if retry_after:
                await _too_many(retry_after)(scope, receive, send)
                return
        await self.app(scope, receive, send)


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
    if not _AUDIT_ADMISSION.acquire(blocking=False):
        log_failure("audit_write_saturated", RuntimeError("audit writer is busy"))
        return
    try:
        future = _AUDIT_EXECUTOR.submit(log_classification, ticket_data, result)
    except Exception as exc:  # noqa: BLE001 - classification is more important
        _AUDIT_ADMISSION.release()
        log_failure("audit_write_failed", exc)
        return
    future.add_done_callback(lambda _future: _AUDIT_ADMISSION.release())
    try:
        await asyncio.wait_for(
            asyncio.wrap_future(future), timeout=AUDIT_WRITE_TIMEOUT_SECONDS
        )
    except TimeoutError:
        log_failure("audit_write_timeout", TimeoutError("audit write timed out"))
    except Exception as exc:  # noqa: BLE001 - audit storage is best effort
        log_failure("audit_write_failed", exc)


async def _generate_with_timeout(
    ticket_data: dict,
    categories: list[str],
    model: str,
    api_url: str,
    timeout: float,
) -> dict:
    """Run one bounded Ollama call and close its client if the caller expires."""
    queue = _QUEUE_SETTINGS
    started = monotonic()
    request_control, reason = await acquire_provider_call(
        queue.max_depth, min(queue.max_wait_seconds, timeout)
    )
    if request_control is None:
        LOGGER.warning("ollama_request_limit_reached (%s)", reason)
        return provider_busy_fallback()
    timeout = max(timeout - (monotonic() - started), 0.001)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                generate_classification,
                ticket_data,
                categories,
                model,
                api_url,
                request_control=request_control,
            ),
            timeout=timeout,
        )
    except (TimeoutError, asyncio.CancelledError):
        request_control.cancel()
        raise
    except Exception:
        request_control.cancel()
        raise
    finally:
        # If the thread never began (for example, it was cancelled in the
        # executor queue), release the admission slot here. A running worker
        # releases it only after closing its client.
        request_control.release_if_not_started()


def _provider_settings(config: Mapping[str, object]) -> tuple[str, str]:
    try:
        _, model, api_url = runtime_config(config)
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
    return {"message": "Admin dashboard not found"}


@app.get("/test")
def test_page():
    test_file = ROOT / "app" / "test.html"
    if test_file.exists():
        return FileResponse(test_file, media_type="text/html")
    return {"message": "Test page not found"}


def _request_config() -> dict:
    """One config generation for the whole request (rules, model and URL)."""
    try:
        return load_config()
    except Exception as exc:
        log_failure("config_load_failed", exc)
        raise HTTPException(
            status_code=503, detail="configuration unavailable"
        ) from exc


def _mark_busy(result: dict, model_out: object) -> dict:
    """Keep the busy marker visible even where rules rewrite the reason."""
    if (
        isinstance(model_out, dict)
        and model_out.get("reason") == BUSY_REASON
        and is_provider_fallback(model_out)
        and not str(result.get("reason", "")).startswith(BUSY_REASON)
    ):
        result["reason"] = f"{BUSY_REASON}; {result.get('reason', '')}"[:2_000]
    return result


@app.post(
    "/classify",
    response_model=ClassificationResponse,
    dependencies=[Depends(require_classify_token)],
)
async def classify(ticket: Ticket):
    # Prompt categories and rule validation share one config snapshot so a
    # concurrent /reload-config cannot make the model return a category the
    # rules engine no longer accepts.
    config = _request_config()
    categories = rules.configured_categories(config)
    model, api_url = _provider_settings(config)
    ticket_data = ticket.model_dump()
    try:
        model_out = await _generate_with_timeout(
            ticket_data, categories, model, api_url, CLASSIFY_TIMEOUT_SECONDS
        )
    except TimeoutError as exc:
        log_failure("classification_timeout", exc)
        model_out = provider_fallback()
    except Exception as exc:
        log_failure("classification_failed", exc)
        raise HTTPException(
            status_code=503, detail="classification unavailable"
        ) from exc
    result = _mark_busy(rules.apply(ticket_data, model_out, config), model_out)
    await _audit_classification(ticket_data, result)
    return ClassificationResponse(**result)


@app.post("/reload-config", dependencies=[Depends(require_admin_token)])
def reload_config():
    try:
        reload_active_config()
        return {"status": "ok"}
    except Exception as exc:
        log_failure("config_reload_failed", exc)
        raise HTTPException(
            status_code=503, detail="configuration unavailable"
        ) from exc


@app.get("/verify-ollama-connection", dependencies=[Depends(require_admin_token)])
def verify_ollama():
    model, api_url = _provider_settings(_request_config())
    check = check_ollama_connection(api_url, model)
    return {
        "model": model,
        "api_url": safe_api_url(api_url),
        **check,
    }


@app.post(
    "/classify-batch",
    response_model=BatchClassificationResponse,
    dependencies=[
        Depends(require_classify_token),
        Depends(rate_limit_batch_remainder),
    ],
)
async def classify_batch(req: BatchClassificationRequest):
    start = monotonic()
    deadline = start + BATCH_TIMEOUT_SECONDS
    results: list[ClassificationResponse] = []
    degraded = 0
    config = _request_config()
    categories = rules.configured_categories(config)
    model, api_url = _provider_settings(config)

    for ticket in req.tickets:
        ticket_data = ticket.model_dump()
        remaining = deadline - monotonic()
        if remaining <= 0:
            result = _unavailable_classification().model_dump()
            degraded += 1
        else:
            try:
                model_out = await _generate_with_timeout(
                    ticket_data,
                    categories,
                    model,
                    api_url,
                    min(BATCH_ITEM_TIMEOUT_SECONDS, remaining),
                )
                result = _mark_busy(
                    rules.apply(ticket_data, model_out, config), model_out
                )
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
        return {"total": count_audit_logs(), "logs": logs, **drop_stats()}
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
app.add_middleware(RateLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
