"""Per-test isolation for the module-global provider gate, limiter and config."""

import sys
import threading
import unittest
from collections import deque
from unittest.mock import patch

from app import main, ollama_client
from app.abuse import ProviderQueueSettings, RateLimitSettings, TokenBucketLimiter


class GateIsolation:
    """Mixin: every test starts with a fresh gate and fails if it leaks one.

    Wraps ``run`` so it works even when a test class defines its own setUp.
    A leaked provider slot or queue position fails the test that caused it
    instead of poisoning every later test in the process.
    """

    def run(self, result=None):
        slots = threading.BoundedSemaphore(ollama_client.MAX_CONCURRENT_PROVIDER_CALLS)
        waiters: deque = deque()
        with (
            patch.object(ollama_client, "_PROVIDER_CALL_SLOTS", slots),
            patch.object(ollama_client, "_WAITERS", waiters),
            patch.object(main, "_LIMITER", TokenBucketLimiter(RateLimitSettings())),
            patch.object(main, "_QUEUE_SETTINGS", ProviderQueueSettings()),
            patch.object(main, "_ACTIVE", main._ACTIVE),
            patch.object(main.rules, "cfg", main.rules.cfg),
        ):
            outcome = super().run(result)
            free = 0
            while slots.acquire(blocking=False):
                free += 1
            problems = []
            if free != ollama_client.MAX_CONCURRENT_PROVIDER_CALLS:
                problems.append(
                    f"provider slots leaked: {free} free of "
                    f"{ollama_client.MAX_CONCURRENT_PROVIDER_CALLS}"
                )
            if waiters:
                problems.append(f"{len(waiters)} queue position(s) leaked")
            if problems and result is not None:
                try:
                    raise AssertionError("; ".join(problems))
                except AssertionError:
                    result.addFailure(self, sys.exc_info())
            return outcome
