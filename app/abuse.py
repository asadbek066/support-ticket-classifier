"""Abuse bounds for the classification endpoints: rate limit and queue settings."""

import math
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import monotonic

MAX_BATCH_TICKETS = 100


@dataclass(frozen=True)
class RateLimitSettings:
    enabled: bool = True
    tickets_per_minute: float = 300.0
    burst: int = 200
    max_clients: int = 10_000


@dataclass(frozen=True)
class ProviderQueueSettings:
    max_depth: int = 32
    max_wait_seconds: float = 30.0


def _section(config: Mapping[str, object], name: str, known: set[str]) -> Mapping:
    raw = config.get(name)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown {name} settings: {sorted(map(str, unknown))}")
    return raw


def _number(raw, section, key, default, low, high, integer=False):
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{section}.{key} must be a number")
    if integer and not isinstance(value, int):
        raise ValueError(f"{section}.{key} must be an integer")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{section}.{key} must be between {low} and {high}")
    return value


def rate_limit_from_config(config: Mapping[str, object]) -> RateLimitSettings:
    raw = _section(
        config,
        "rate_limit",
        {"enabled", "tickets_per_minute", "burst", "max_clients"},
    )
    defaults = RateLimitSettings()
    enabled = raw.get("enabled", defaults.enabled)
    if not isinstance(enabled, bool):
        raise ValueError("rate_limit.enabled must be true or false")
    return RateLimitSettings(
        enabled=enabled,
        tickets_per_minute=float(
            _number(
                raw, "rate_limit", "tickets_per_minute",
                defaults.tickets_per_minute, 1, 1_000_000,
            )
        ),
        # A full batch is charged per ticket, so the bucket must hold one.
        burst=_number(
            raw, "rate_limit", "burst", defaults.burst,
            MAX_BATCH_TICKETS, 1_000_000, True,
        ),
        max_clients=_number(
            raw, "rate_limit", "max_clients", defaults.max_clients, 1, 1_000_000, True
        ),
    )


def provider_queue_from_config(config: Mapping[str, object]) -> ProviderQueueSettings:
    raw = _section(config, "provider_queue", {"max_depth", "max_wait_seconds"})
    defaults = ProviderQueueSettings()
    return ProviderQueueSettings(
        max_depth=_number(
            raw, "provider_queue", "max_depth", defaults.max_depth, 0, 10_000, True
        ),
        max_wait_seconds=float(
            _number(
                raw, "provider_queue", "max_wait_seconds",
                defaults.max_wait_seconds, 0, 80,
            )
        ),
    )


class TokenBucketLimiter:
    """In-process per-client token bucket; a request is charged ``cost`` tokens."""

    def __init__(
        self,
        settings: RateLimitSettings,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.settings = settings
        self._clock = clock
        self._rate = settings.tickets_per_minute / 60.0
        self._lock = threading.Lock()
        # key -> [tokens, last_refill]; ordered least recently used first.
        self._buckets: OrderedDict[str, list[float]] = OrderedDict()

    def check(self, key: str, cost: int = 1) -> int:
        """Charge the client; return 0 if allowed, else seconds to retry."""
        if not self.settings.enabled:
            return 0
        capacity = float(self.settings.burst)
        with self._lock:
            now = self._clock()
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = [capacity, now]
                self._buckets[key] = bucket
                self._evict(now)
            else:
                elapsed = max(0.0, now - bucket[1])
                bucket[0] = min(capacity, bucket[0] + elapsed * self._rate)
                bucket[1] = now
                self._buckets.move_to_end(key)
            if cost <= bucket[0]:
                bucket[0] -= cost
                return 0
            return max(1, math.ceil((min(cost, capacity) - bucket[0]) / self._rate))

    def _evict(self, now: float) -> None:
        while len(self._buckets) > self.settings.max_clients:
            self._buckets.popitem(last=False)
