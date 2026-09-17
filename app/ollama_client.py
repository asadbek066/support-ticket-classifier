import json
import logging
from time import monotonic
from urllib.parse import urlsplit

import httpx

LOGGER = logging.getLogger(__name__)
FALLBACK_REASON = "model-unreachable-or-invalid-response"
FALLBACK_MARKER = "_provider_fallback"
MAX_RESPONSE_CHARS = 20_000
MAX_STREAM_SECONDS = 75.0
CONNECTION_CHECK_TIMEOUT_SECONDS = 5.0
GENERATION_TIMEOUT_SECONDS = 60.0


def provider_fallback() -> dict:
    return {
        "category": "Other / Needs Review",
        "confidence": 0.0,
        "queue": "triage",
        "reason": FALLBACK_REASON,
        "human_review": True,
        FALLBACK_MARKER: True,
    }


def is_provider_fallback(model_out: object) -> bool:
    return (
        isinstance(model_out, dict)
        and model_out.get(FALLBACK_MARKER) is True
    )


def _read_bounded_response(response, deadline: float) -> str | None:
    content_length = response.headers.get("content-length")
    try:
        if content_length is not None and int(content_length) > MAX_RESPONSE_CHARS:
            return None
    except (TypeError, ValueError):
        pass

    chunks = []
    total_bytes = 0
    for chunk in response.iter_bytes():
        # httpx read timeouts apply per chunk, so a slow trickle could stretch
        # indefinitely; enforce a wall-clock bound on the whole stream.
        if monotonic() > deadline:
            LOGGER.warning("ollama_stream_deadline_exceeded")
            return None
        total_bytes += len(chunk)
        if total_bytes > MAX_RESPONSE_CHARS:
            return None
        chunks.append(chunk)
    try:
        return b"".join(chunks).decode(response.encoding or "utf-8")
    except (AttributeError, UnicodeDecodeError):
        return None


def generate_classification(ticket: dict, categories, model: str, api_url: str) -> dict:
    prompt = build_prompt(ticket, categories)
    payload = {"model": model, "prompt": prompt, "stream": False, "temperature": 0}
    deadline = monotonic() + MAX_STREAM_SECONDS
    try:
        with httpx.stream(
            "POST", api_url, json=payload, timeout=GENERATION_TIMEOUT_SECONDS
        ) as response:
            response.raise_for_status()
            text = _read_bounded_response(response, deadline)
    except Exception as exc:  # noqa: BLE001
        # The local model boundary must never leak provider diagnostics or
        # secrets; log only the failure class for operations.
        LOGGER.warning("ollama_request_failed (%s)", type(exc).__name__)
        return provider_fallback()

    if text is None:
        LOGGER.warning("ollama_response_rejected")
        return provider_fallback()
    text = text.strip()
    if len(text) > MAX_RESPONSE_CHARS:
        return provider_fallback()
    lines = text.split("\n")

    if len(lines) > 1:
        final_response = None
        for line in reversed(lines):
            if line.strip():
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and obj.get("done", False):
                        candidate = obj.get("response", "")
                        final_response = candidate if isinstance(candidate, str) else ""
                        break
                except (AttributeError, TypeError, ValueError):
                    pass
        if final_response is not None:
            text = final_response
        else:
            try:
                last_obj = json.loads(lines[-1].strip())
                if isinstance(last_obj, dict) and isinstance(
                    last_obj.get("response"), str
                ):
                    text = last_obj["response"]
            except (AttributeError, TypeError, ValueError):
                pass
    else:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                candidate = obj.get("response", "") or obj.get("text", "") or text
                if isinstance(candidate, str):
                    text = candidate
        except (AttributeError, TypeError, ValueError):
            pass

    if not isinstance(text, str) or len(text) > MAX_RESPONSE_CHARS:
        return provider_fallback()

    # Equivalent to a greedy "\{.*\}" search but linear: a long run of "{"
    # without a closing brace must not trigger quadratic backtracking.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        last = text.strip().splitlines()[-1] if text.strip() else ""
        try:
            parsed = json.loads(last)
            return parsed if isinstance(parsed, dict) else provider_fallback()
        except (AttributeError, TypeError, ValueError):
            return provider_fallback()

    try:
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, dict) else provider_fallback()
    except (AttributeError, TypeError, ValueError):
        return provider_fallback()


def _tags_endpoint(api_url: str) -> str:
    parsed = urlsplit(api_url)
    path = parsed.path.removesuffix("/api/generate")
    return f"{parsed.scheme}://{parsed.netloc}{path}/api/tags"


def _model_is_available(models: object, model: str) -> bool:
    if not isinstance(models, list):
        return False
    configured = model.strip()
    normalized = configured if ":" in configured else f"{configured}:latest"
    for entry in models:
        if not isinstance(entry, dict):
            continue
        names = {entry.get("name"), entry.get("model")}
        if configured in names or normalized in names:
            return True
    return False


def check_ollama_connection(api_url: str, model: str) -> dict:
    """Probe server liveness and model presence without running inference."""
    try:
        response = httpx.get(
            _tags_endpoint(api_url),
            timeout=CONNECTION_CHECK_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 - any transport/config error degrades
        LOGGER.warning("ollama_check_failed (%s)", type(exc).__name__)
        return {
            "connected": False,
            "error": "Ollama is unavailable",
            "hint": "Is Ollama running? Check: (1) Start Ollama desktop app, (2) Or run 'ollama serve' in a terminal, (3) Check if api_url in config.yaml is correct.",
        }
    if response.status_code != 200:
        return {
            "connected": False,
            "status_code": response.status_code,
            "error": "Ollama returned an error",
        }
    try:
        payload = response.json()
        models = payload.get("models") if isinstance(payload, dict) else None
    except ValueError:
        models = None
    return {
        "connected": True,
        "model_available": _model_is_available(models, model),
    }


def build_prompt(ticket: dict, categories) -> str:
    cat_list = "; ".join(categories)
    prompt = (
        "You are a ticket classification assistant.\n"
        "Given the ticket fields, return STRICT JSON with keys: category, confidence (0-1), queue, reason, human_review.\n"
        "Only output valid JSON and nothing else.\n\n"
        f"Categories: {cat_list}\n\n"
        "Example:\n"
        "Ticket: subject='Payment failed', description='My credit card was declined while paying invoice', source_channel='email', customer_type='paid'\n"
        'Output: {\n  "category": "Billing & Payments",\n  "confidence": 0.95,\n  "queue": "billing-queue",\n  "reason": "Payment failure language and billing keywords",\n  "human_review": false\n}\n\n'
        "Now classify this ticket:\n"
        f"Ticket: subject={json.dumps(ticket.get('subject', ''))}, description={json.dumps(ticket.get('description', ''))}, source_channel={json.dumps(ticket.get('source_channel', ''))}, customer_type={json.dumps(ticket.get('customer_type', ''))}, language={json.dumps(ticket.get('language', ''))}\n"
    )
    return prompt
