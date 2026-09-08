import json
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

AUDIT_LOG_DIR = Path(__file__).resolve().parents[1] / "audit_logs"
MAX_AUDIT_LINE_CHARS = 128 * 1024


def ensure_audit_dir() -> None:
    AUDIT_LOG_DIR.mkdir(exist_ok=True)


def log_classification(ticket: dict, result: dict, overrides: dict | None = None) -> str:
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


def read_audit_logs(limit: int = 100) -> list[dict]:
    limit = min(max(limit, 1), 1000)
    ensure_audit_dir()
    entries = []
    log_files = sorted(AUDIT_LOG_DIR.glob("classifications-*.jsonl"), reverse=True)
    for log_file in log_files:
        recent_entries = deque(maxlen=limit)
        with open(log_file, "r", encoding="utf-8") as f:
            while True:
                line = f.readline(MAX_AUDIT_LINE_CHARS + 1)
                if not line:
                    break
                if len(line) > MAX_AUDIT_LINE_CHARS and not line.endswith("\n"):
                    while line and not line.endswith("\n"):
                        line = f.readline(MAX_AUDIT_LINE_CHARS + 1)
                    continue
                if len(line) > MAX_AUDIT_LINE_CHARS:
                    continue
                if line.strip():
                    try:
                        value = json.loads(line)
                        if isinstance(value, dict):
                            recent_entries.append(value)
                    except json.JSONDecodeError:
                        pass
        entries.extend(reversed(recent_entries))
        if len(entries) >= limit:
            return entries[:limit]
    return entries


def get_low_confidence_tickets(threshold: float = 0.65, limit: int = 50) -> list[dict]:
    all_entries = read_audit_logs(limit=1000)
    low_conf = [e for e in all_entries if e.get("classification", {}).get("confidence", 1.0) < threshold]
    return low_conf[:limit]
