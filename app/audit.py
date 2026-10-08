import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from time import monotonic

LOGGER = logging.getLogger(__name__)

AUDIT_LOG_DIR = Path(__file__).resolve().parents[1] / "audit_logs"
MAX_AUDIT_LINE_BYTES = 128 * 1024
TAIL_CHUNK_BYTES = 64 * 1024
FINGERPRINT_BYTES = 64
MAX_CACHED_THRESHOLDS = 16
AUDIT_DIRECTORY_MODE = 0o700
AUDIT_FILE_MODE = 0o600

DEFAULT_KEY_PATH = Path(__file__).resolve().parents[1] / "audit_hmac.key"
DIGEST_SCHEME = "hmac-sha256"
AUDIT_FILE_RE = re.compile(r"classifications-(\d{4}-\d{2}-\d{2})\.jsonl")
MEGABYTE = 1024 * 1024
TICKET_TEXT_LIMIT = 20_000


@dataclass(frozen=True)
class AuditSettings:
    """Bounds for what the audit log stores and how long it keeps it."""

    max_text_chars: int = 300
    store_full_text: bool = False
    retention_days: int = 30
    max_total_bytes: int = 200 * MEGABYTE
    max_daily_bytes: int = 20 * MEGABYTE
    prune_interval_seconds: float = 3600.0
    hash_key_file: str | None = None  # None: DEFAULT_KEY_PATH


def settings_from_config(config: Mapping[str, object]) -> AuditSettings:
    """Build validated settings from the optional ``audit`` config section.

    Raises ValueError for any invalid value so a bad edit is rejected as a
    whole rather than silently weakening a bound.
    """
    raw = config.get("audit")
    if raw is None:
        return AuditSettings()
    if not isinstance(raw, Mapping):
        raise ValueError("audit must be a mapping")
    known = {
        "max_text_chars",
        "store_full_text",
        "retention_days",
        "max_total_mb",
        "max_daily_mb",
        "prune_interval_seconds",
        "hash_key_file",
    }
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown audit settings: {sorted(map(str, unknown))}")
    defaults = AuditSettings()

    def number(key, default, low, high, integer=False):
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"audit.{key} must be a number")
        if integer and not isinstance(value, int):
            raise ValueError(f"audit.{key} must be an integer")
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"audit.{key} must be between {low} and {high}")
        return value

    store_full = raw.get("store_full_text", defaults.store_full_text)
    if not isinstance(store_full, bool):
        raise ValueError("audit.store_full_text must be true or false")
    key_file = raw.get("hash_key_file")
    if key_file is not None and (
        not isinstance(key_file, str) or not key_file.strip() or len(key_file) > 500
    ):
        raise ValueError("audit.hash_key_file must be a non-empty path string")
    return AuditSettings(
        hash_key_file=key_file,
        max_text_chars=number(
            "max_text_chars", defaults.max_text_chars, 0, TICKET_TEXT_LIMIT, True
        ),
        store_full_text=store_full,
        retention_days=number(
            "retention_days", defaults.retention_days, 1, 3650, True
        ),
        max_total_bytes=int(
            number("max_total_mb", defaults.max_total_bytes / MEGABYTE, 1, 1_000_000)
            * MEGABYTE
        ),
        max_daily_bytes=int(
            number("max_daily_mb", defaults.max_daily_bytes / MEGABYTE, 1, 100_000)
            * MEGABYTE
        ),
        prune_interval_seconds=float(
            number(
                "prune_interval_seconds",
                defaults.prune_interval_seconds,
                1,
                30 * 86400,
            )
        ),
    )


_SETTINGS = AuditSettings()
_STATE_LOCK = threading.Lock()
_DROPPED_BY_DAY: dict[str, int] = {}
_DROPPED_TOTAL = 0
_WARNED_DAYS: set[str] = set()
_LAST_PRUNE: float | None = None


def configure(settings: AuditSettings) -> None:
    """Swap the active settings atomically (called at startup and on reload)."""
    global _SETTINGS
    _SETTINGS = settings


def get_settings() -> AuditSettings:
    return _SETTINGS


def drop_stats(now: datetime | None = None) -> dict:
    """Records dropped since process start because the daily cap was reached."""
    day = (now or datetime.now(UTC)).strftime("%Y-%m-%d")
    with _STATE_LOCK:
        return {
            "dropped_today": _DROPPED_BY_DAY.get(day, 0),
            "dropped_total": _DROPPED_TOTAL,
        }


def _record_drop(day: str) -> None:
    global _DROPPED_TOTAL
    with _STATE_LOCK:
        _DROPPED_BY_DAY[day] = _DROPPED_BY_DAY.get(day, 0) + 1
        _DROPPED_TOTAL += 1
        # Keep the per-day map from growing for the life of the process.
        for old in sorted(_DROPPED_BY_DAY)[:-7]:
            del _DROPPED_BY_DAY[old]
        first = day not in _WARNED_DAYS
        if first:
            _WARNED_DAYS.add(day)
            for old in sorted(_WARNED_DAYS)[:-7]:
                _WARNED_DAYS.discard(old)
    if first:
        LOGGER.warning(
            "audit_daily_cap_reached: further records for %s are dropped "
            "(classification continues)",
            day,
        )


_KEY_LOCK = threading.Lock()
_KEY_CACHE: dict[Path, bytes | None] = {}
_KEY_WARNED: set[Path] = set()


def _key_path(settings: AuditSettings) -> Path:
    if settings.hash_key_file is None:
        return DEFAULT_KEY_PATH
    path = Path(settings.hash_key_file).expanduser()
    return path if path.is_absolute() else DEFAULT_KEY_PATH.parent / path


def _load_or_create_key(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        descriptor = -1
    if descriptor < 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        new = secrets.token_hex(32)
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                AUDIT_FILE_MODE,
            )
        except FileExistsError:  # another worker created it first
            return _load_or_create_key(path)
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(new + "\n")
        return bytes.fromhex(new)
    with os.fdopen(descriptor, "r", encoding="ascii") as handle:
        value = handle.read().strip()
    key = bytes.fromhex(value)
    if len(key) < 32:
        raise ValueError("audit hash key is too short")
    return key


def _digest_key(settings: AuditSettings) -> bytes | None:
    """Per-installation HMAC key, or None (logged once) if unavailable."""
    path = _key_path(settings)
    with _KEY_LOCK:
        if path in _KEY_CACHE:
            return _KEY_CACHE[path]
        try:
            directory = AUDIT_LOG_DIR.resolve()
            if path.resolve().is_relative_to(directory):
                raise ValueError("audit hash key must live outside the audit directory")
            key = _load_or_create_key(path)
        except (OSError, ValueError) as exc:
            key = None
            if path not in _KEY_WARNED:
                _KEY_WARNED.add(path)
                LOGGER.error(
                    "audit_hash_key_unavailable (%s); ticket digests are omitted",
                    type(exc).__name__,
                )
        # A failed lookup is not cached, so a fixed key file is picked up.
        if key is not None:
            _KEY_CACHE[path] = key
        return key


def _bounded_ticket(ticket: dict, settings: AuditSettings) -> dict:
    """Return the ticket as stored: text truncated, with keyed hash and length."""
    if settings.store_full_text:
        stored = dict(ticket)
    else:
        stored = {}
    secret = _digest_key(settings)
    digests: dict[str, dict] = {}
    truncated: list[str] = []
    for field, value in ticket.items():
        if not isinstance(value, str):
            if not settings.store_full_text:
                stored[field] = value
            continue
        digests[field] = {"chars": len(value)}
        if secret is not None:
            digests[field]["hmac"] = hmac.new(
                secret, value.encode("utf-8"), hashlib.sha256
            ).hexdigest()
        if not settings.store_full_text:
            if len(value) > settings.max_text_chars:
                truncated.append(field)
            stored[field] = value[: settings.max_text_chars]
    record = {
        "ticket": stored,
        "ticket_digest": digests,
        "truncated_fields": truncated,
    }
    if secret is not None:
        record["digest_scheme"] = DIGEST_SCHEME
    return record


def _audit_candidates(directory: Path) -> list[tuple[date, str, int]]:
    """Regular files in the audit dir matching the filename pattern.

    Symlinks and everything else are ignored, never followed or deleted.
    """
    found = []
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return found
    for entry in entries:
        match = AUDIT_FILE_RE.fullmatch(entry.name)
        if match is None:
            continue
        try:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                continue
            file_date = date.fromisoformat(match.group(1))
            size = entry.stat(follow_symlinks=False).st_size
        except (OSError, ValueError):
            continue
        found.append((file_date, entry.name, size))
    return sorted(found)


def prune_audit_files(
    settings: AuditSettings | None = None,
    now: datetime | None = None,
    current_name: str | None = None,
) -> list[str]:
    """Delete expired files, then oldest files until under the size cap."""
    settings = settings or _SETTINGS
    now = now or datetime.now(UTC)
    if current_name is None:
        current_name = f"classifications-{now.strftime('%Y-%m-%d')}.jsonl"
    directory = AUDIT_LOG_DIR
    if not _safe_to_prune(directory):
        return []
    cutoff = now.date() - timedelta(days=settings.retention_days)
    removed: list[str] = []
    remaining = []
    for file_date, name, size in _audit_candidates(directory):
        if name != current_name and file_date < cutoff:
            if _unlink(directory / name):
                removed.append(name)
                continue
        remaining.append((file_date, name, size))
    total = sum(size for _, _, size in remaining)
    for _, name, size in remaining:
        if total <= settings.max_total_bytes:
            break
        if name == current_name:
            continue
        if _unlink(directory / name):
            removed.append(name)
            total -= size
    if removed:
        with _STATS_LOCK:
            for name in removed:
                _FILE_STATS.pop(directory / name, None)
        LOGGER.info("audit_pruned files=%d", len(removed))
    return removed


_UNSAFE_WARNED: set[Path] = set()


def _safe_to_prune(directory: Path) -> bool:
    """Only prune a real directory owned by this user (writes are unaffected)."""
    try:
        if directory.is_symlink():
            reason = "is a symlink"
        elif hasattr(os, "getuid") and os.stat(directory).st_uid != os.getuid():
            reason = "is not owned by the service user"
        else:
            return True
    except OSError:
        return False
    with _STATE_LOCK:
        first = directory not in _UNSAFE_WARNED
        _UNSAFE_WARNED.add(directory)
    if first:
        LOGGER.error("audit_prune_refused: audit directory %s", reason)
    return False


def _unlink(path: Path) -> bool:
    try:
        os.unlink(path)  # unlink never follows a symlink target
        return True
    except OSError as exc:
        LOGGER.warning("audit_prune_failed (%s)", type(exc).__name__)
        return False


def _maybe_prune(settings: AuditSettings, current_name: str) -> None:
    """Prune at most once per interval; the first call in a process runs."""
    global _LAST_PRUNE
    with _STATE_LOCK:
        moment = monotonic()
        if (
            _LAST_PRUNE is not None
            and moment - _LAST_PRUNE < settings.prune_interval_seconds
        ):
            return
        _LAST_PRUNE = moment
    try:
        prune_audit_files(settings, current_name=current_name)
    except Exception as exc:  # noqa: BLE001 - never fail a write over housekeeping
        LOGGER.warning("audit_prune_failed (%s)", type(exc).__name__)


def startup_prune() -> None:
    """Prune once at service start and restart the interval clock."""
    global _LAST_PRUNE
    with _STATE_LOCK:
        _LAST_PRUNE = monotonic()
    prune_audit_files(_SETTINGS)


# Per-file incremental counters. Audit files are append-only, so counts are
# advanced by parsing only the bytes appended since the last call. The lock
# makes the cache safe for the threadpool that runs the sync endpoints.
_STATS_LOCK = threading.Lock()
_FILE_STATS: dict[Path, dict] = {}


def ensure_audit_dir() -> None:
    AUDIT_LOG_DIR.mkdir(mode=AUDIT_DIRECTORY_MODE, exist_ok=True)
    if os.name == "posix":
        os.chmod(AUDIT_LOG_DIR, AUDIT_DIRECTORY_MODE)


def log_classification(
    ticket: dict, result: dict, overrides: dict | None = None
) -> str:
    ensure_audit_dir()
    settings = _SETTINGS
    now = datetime.now(UTC)
    entry = {
        "timestamp": now.isoformat(),
        **_bounded_ticket(ticket, settings),
        "classification": result,
        "overrides": overrides or {},
    }
    day = now.strftime("%Y-%m-%d")
    log_file = AUDIT_LOG_DIR / f"classifications-{day}.jsonl"
    if log_file.is_symlink():
        raise OSError("audit log path must not be a symlink")
    line = json.dumps(entry) + "\n"
    descriptor = os.open(
        log_file,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_APPEND
        | getattr(os, "O_NOFOLLOW", 0),
        AUDIT_FILE_MODE,
    )
    try:
        if os.name == "posix":
            os.fchmod(descriptor, AUDIT_FILE_MODE)
        if os.fstat(descriptor).st_size >= settings.max_daily_bytes:
            _record_drop(day)
        else:
            with os.fdopen(descriptor, "a", encoding="utf-8") as f:
                descriptor = -1
                f.write(line)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _maybe_prune(settings, log_file.name)
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
