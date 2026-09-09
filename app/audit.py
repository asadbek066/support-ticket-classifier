import json
import math
from collections import deque
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

AUDIT_LOG_DIR = Path(__file__).resolve().parents[1] / "audit_logs"
MAX_AUDIT_LINE_CHARS = 128 * 1024


def ensure_audit_dir() -> None:
    AUDIT_LOG_DIR.mkdir(exist_ok=True)


def log_classification(
    ticket: dict, result: dict, overrides: dict | None = None
) -> str:
    ensure_audit_dir()
    timestamp = datetime.now(UTC).isoformat()
    entry = {
        "timestamp": timestamp,
        "ticket": ticket,
        "classification": result,
        "overrides": overrides or {},
    }
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    log_file = AUDIT_LOG_DIR / f"classifications-{today}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return log_file.name


def _valid_entries(log_file: Path) -> Iterator[dict]:
    with open(log_file, "r", encoding="utf-8") as f:
        while True:
            line = f.readline(MAX_AUDIT_LINE_CHARS + 1)
            if not line:
                break
            if len(line) > MAX_AUDIT_LINE_CHARS and not line.endswith("\n"):
                while line and not line.endswith("\n"):
                    line = f.readline(MAX_AUDIT_LINE_CHARS + 1)
                continue
            if len(line) > MAX_AUDIT_LINE_CHARS or not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def read_audit_logs(limit: int = 100) -> list[dict]:
    limit = min(max(limit, 1), 1000)
    ensure_audit_dir()
    entries: list[dict] = []
    log_files = sorted(AUDIT_LOG_DIR.glob("classifications-*.jsonl"), reverse=True)
    for log_file in log_files:
        recent_entries: deque[dict] = deque(maxlen=limit)
        recent_entries.extend(_valid_entries(log_file))
        entries.extend(reversed(recent_entries))
        if len(entries) >= limit:
            return entries[:limit]
    return entries


def count_audit_logs() -> int:
    ensure_audit_dir()
    return sum(
        1
        for log_file in AUDIT_LOG_DIR.glob("classifications-*.jsonl")
        for _ in _valid_entries(log_file)
    )


def _requires_review(entry: dict, threshold: float) -> bool:
    classification = entry.get("classification")
    if not isinstance(classification, dict):
        return False
    if classification.get("human_review") is True:
        return True
    confidence = classification.get("confidence")
    return (
        not isinstance(confidence, bool)
        and isinstance(confidence, (int, float))
        and math.isfinite(confidence)
        and confidence < threshold
    )


def get_low_confidence_tickets(threshold: float = 0.65, limit: int = 50) -> list[dict]:
    limit = min(max(limit, 1), 1000)
    if not math.isfinite(threshold):
        threshold = 0.65

    review_entries: list[dict] = []
    log_files = sorted(AUDIT_LOG_DIR.glob("classifications-*.jsonl"), reverse=True)
    for log_file in log_files:
        newest_matches: deque[dict] = deque(maxlen=limit)
        for entry in _valid_entries(log_file):
            if _requires_review(entry, threshold):
                newest_matches.append(entry)
        review_entries.extend(reversed(newest_matches))
        if len(review_entries) >= limit:
            return review_entries[:limit]
    return review_entries
