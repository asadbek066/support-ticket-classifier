import json
import math
import threading
from datetime import UTC, datetime
from pathlib import Path

AUDIT_LOG_DIR = Path(__file__).resolve().parents[1] / "audit_logs"
MAX_AUDIT_LINE_BYTES = 128 * 1024
TAIL_CHUNK_BYTES = 64 * 1024
FINGERPRINT_BYTES = 64
MAX_CACHED_THRESHOLDS = 16

# Per-file incremental counters. Audit files are append-only, so counts are
# advanced by parsing only the bytes appended since the last call. The lock
# makes the cache safe for the threadpool that runs the sync endpoints.
_STATS_LOCK = threading.Lock()
_FILE_STATS: dict[Path, dict] = {}


def ensure_audit_dir() -> None:
    AUDIT_LOG_DIR.mkdir(exist_ok=True)


def log_classification(
    ticket: dict, result: dict, overrides: dict | None = None
) -> str:
    ensure_audit_dir()
    now = datetime.now(UTC)
    entry = {
        "timestamp": now.isoformat(),
        "ticket": ticket,
        "classification": result,
        "overrides": overrides or {},
    }
    log_file = AUDIT_LOG_DIR / f"classifications-{now.strftime('%Y-%m-%d')}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return log_file.name


def _parse_raw_line(raw: bytes) -> dict | None:
    if len(raw) > MAX_AUDIT_LINE_BYTES or not raw.strip():
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _scan_entries(
    path: Path, offset: int, thresholds: tuple[float, ...]
) -> tuple[int, int, dict[float, int]]:
    """Parse valid entries from offset onward; return offset, count, matches."""
    valid = 0
    matches = {threshold: 0 for threshold in thresholds}
    with open(path, "rb") as f:
        f.seek(offset)
        while True:
            line_start = f.tell()
            raw = f.readline(MAX_AUDIT_LINE_BYTES + 1)
            if not raw:
                break
            if not raw.endswith(b"\n"):
                if len(raw) <= MAX_AUDIT_LINE_BYTES:
                    # A torn trailing write; leave the offset before it so the
                    # completed line is counted on the next pass.
                    return line_start, valid, matches
                while raw and not raw.endswith(b"\n"):
                    raw = f.readline(MAX_AUDIT_LINE_BYTES + 1)
                continue
            if len(raw) > MAX_AUDIT_LINE_BYTES:
                continue
            entry = _parse_raw_line(raw)
            if entry is None:
                continue
            valid += 1
            for threshold in thresholds:
                if _requires_review(entry, threshold):
                    matches[threshold] += 1
        return f.tell(), valid, matches


def _tail_entries(
    path: Path, limit: int, threshold: float | None = None
) -> list[dict]:
    """Return up to limit valid entries from the end of the file, newest first."""
    found: list[dict] = []
    with open(path, "rb") as f:
        f.seek(0, 2)
        position = f.tell()
        tail = b""
        while position > 0 and len(found) < limit:
            read_size = min(TAIL_CHUNK_BYTES, position)
            position -= read_size
            f.seek(position)
            chunk = f.read(read_size) + tail
            parts = chunk.split(b"\n")
            if position > 0:
                tail = parts[0]
                complete = parts[1:]
            else:
                tail = b""
                complete = parts
            for raw in reversed(complete):
                entry = _parse_raw_line(raw)
                if entry is None:
                    continue
                if threshold is not None and not _requires_review(entry, threshold):
                    continue
                found.append(entry)
                if len(found) >= limit:
                    break
    return found


def _audit_files() -> list[Path]:
    return sorted(AUDIT_LOG_DIR.glob("classifications-*.jsonl"), reverse=True)


def read_audit_logs(limit: int = 100) -> list[dict]:
    limit = min(max(limit, 1), 1000)
    ensure_audit_dir()
    entries: list[dict] = []
    for log_file in _audit_files():
        entries.extend(_tail_entries(log_file, limit - len(entries)))
        if len(entries) >= limit:
            return entries[:limit]
    return entries


def _read_fingerprint(path: Path, offset: int) -> bytes:
    """Bytes immediately before offset, used to detect in-place rewrites."""
    if offset <= 0:
        return b""
    start = max(0, offset - FINGERPRINT_BYTES)
    try:
        with open(path, "rb") as f:
            f.seek(start)
            return f.read(offset - start)
    except OSError:
        return b""


def _file_stats(path: Path, thresholds: tuple[float, ...]) -> tuple[int, dict]:
    with _STATS_LOCK:
        stats = _FILE_STATS.get(path)
        if stats is None:
            stats = {
                "identity": None,
                "mtime_ns": None,
                "valid_offset": 0,
                "valid": 0,
                "thresholds": {},
                "fingerprint": b"",
            }
            _FILE_STATS[path] = stats
        try:
            file_stat = path.stat()
        except OSError:
            return 0, {}
        size = file_stat.st_size
        identity = (file_stat.st_dev, file_stat.st_ino)
        if (
            stats["identity"] != identity
            or size < stats["valid_offset"]
            or (
                stats["valid_offset"] > 0
                and _read_fingerprint(path, stats["valid_offset"])
                != stats["fingerprint"]
            )
            or (
                size == stats["valid_offset"]
                and stats["mtime_ns"] is not None
                and file_stat.st_mtime_ns != stats["mtime_ns"]
            )
        ):
            # The file was replaced, truncated, or rewritten in place (for
            # example copytruncate rotation); recount from the start.
            stats["valid_offset"] = 0
            stats["valid"] = 0
            stats["thresholds"] = {}
            stats["fingerprint"] = b""
        stats["identity"] = identity
        stats["mtime_ns"] = file_stat.st_mtime_ns
        if size != stats["valid_offset"]:
            offset, valid, _ = _scan_entries(path, stats["valid_offset"], ())
            stats["valid_offset"] = offset
            stats["valid"] += valid
            stats["fingerprint"] = _read_fingerprint(path, offset)
        for threshold in thresholds:
            cached = stats["thresholds"].get(threshold)
            if cached is not None and cached[0] == stats["valid_offset"]:
                continue
            start = cached[0] if cached is not None else 0
            count = cached[1] if cached is not None else 0
            offset, _, matches = _scan_entries(path, start, (threshold,))
            stats["thresholds"][threshold] = (offset, count + matches[threshold])
        if thresholds and len(stats["thresholds"]) > MAX_CACHED_THRESHOLDS:
            stats["thresholds"] = {
                threshold: stats["thresholds"][threshold]
                for threshold in thresholds
                if threshold in stats["thresholds"]
            }
        return stats["valid"], {
            threshold: stats["thresholds"].get(threshold, (0, 0))[1]
            for threshold in thresholds
        }


def count_audit_logs() -> int:
    ensure_audit_dir()
    return sum(_file_stats(log_file, ())[0] for log_file in _audit_files())


def count_low_confidence_tickets(threshold: float = 0.65) -> int:
    if not math.isfinite(threshold):
        threshold = 0.65
    ensure_audit_dir()
    return sum(
        _file_stats(log_file, (threshold,))[1].get(threshold, 0)
        for log_file in _audit_files()
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
    ensure_audit_dir()
    review_entries: list[dict] = []
    for log_file in _audit_files():
        review_entries.extend(
            _tail_entries(log_file, limit - len(review_entries), threshold)
        )
        if len(review_entries) >= limit:
            return review_entries[:limit]
    return review_entries
