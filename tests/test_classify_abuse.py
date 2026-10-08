import asyncio
import functools
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main, ollama_client
from app.abuse import (
    ProviderQueueSettings,
    RateLimitSettings,
    TokenBucketLimiter,
    provider_queue_from_config,
    rate_limit_from_config,
)
from app.main import app
from app.ollama_client import (
    MAX_CONCURRENT_PROVIDER_CALLS,
    ProviderCall,
    acquire_provider_call,
    queued_provider_waiters,
)

GOOD = {"category": "Billing & Payments", "confidence": 0.9, "reason": "ok",
        "human_review": False}


def post_one(client, **kwargs):
    return client.post("/classify", json={"subject": "Payment failed"}, **kwargs)


SCENARIO_TIMEOUT = 5.0


def bounded(test):
    """Fail fast instead of hanging if a queue regression strands a waiter."""

    @functools.wraps(test)
    async def wrapper(self):
        await asyncio.wait_for(test(self), SCENARIO_TIMEOUT)

    return wrapper


class RateLimitEndpointTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, main, "_LIMITER", main._LIMITER)
        main._LIMITER = TokenBucketLimiter(
            RateLimitSettings(tickets_per_minute=60, burst=100)
        )
        for item in (
            patch("app.main.generate_classification", return_value=GOOD),
            patch("app.main.log_classification"),
            # Never queue here: a leaked slot must fail fast, not wait 30s per call.
            patch.object(main, "_QUEUE_SETTINGS", ProviderQueueSettings(0, 0.0)),
        ):
            item.start()
            self.addCleanup(item.stop)

    def client(self, host):
        return TestClient(app, client=(host, 5000))

    def test_limit_is_per_client_and_returns_retry_after(self):
        main._LIMITER = TokenBucketLimiter(
            RateLimitSettings(tickets_per_minute=60, burst=100),
            clock=lambda: 0.0,
        )
        noisy, quiet = self.client("10.0.0.1"), self.client("10.0.0.2")
        for _ in range(100):
            self.assertEqual(post_one(noisy).status_code, 200)
        blocked = post_one(noisy)
        self.assertEqual(blocked.status_code, 429)
        self.assertEqual(blocked.headers["retry-after"], "1")
        self.assertEqual(post_one(quiet).status_code, 200)

    def test_batch_is_charged_per_ticket(self):
        main._LIMITER = TokenBucketLimiter(
            RateLimitSettings(tickets_per_minute=60, burst=100),
            clock=lambda: 0.0,
        )
        client = self.client("10.0.0.3")
        batch = {"tickets": [{"subject": f"s{i}"} for i in range(60)]}
        self.assertEqual(client.post("/classify-batch", json=batch).status_code, 200)
        denied = client.post("/classify-batch", json=batch)
        self.assertEqual(denied.status_code, 429)
        self.assertEqual(denied.headers["retry-after"], "20")  # 20 tokens / 1 per s
        # The denied batch still cost its up-front token: 39 remain, so a
        # 39-ticket batch passes and then nothing is left.
        small = {"tickets": [{"subject": "x"}] * 39}
        self.assertEqual(client.post("/classify-batch", json=small).status_code, 200)
        self.assertEqual(post_one(client).status_code, 429)

    def test_invalid_bodies_are_charged_on_both_endpoints(self):
        bad_bodies = {
            "/classify": [
                {"json": {"subject": 5}},
                {"content": b"{not json", "headers": {"Content-Type": "application/json"}},
            ],
            "/classify-batch": [
                {"json": {"tickets": []}},
                {"content": b"{not json", "headers": {"Content-Type": "application/json"}},
            ],
        }
        for path, variants in bad_bodies.items():
            for kwargs in variants:
                main._LIMITER = TokenBucketLimiter(
                    RateLimitSettings(tickets_per_minute=60, burst=100),
                    clock=lambda: 0.0,
                )
                client = self.client("10.0.0.7")
                statuses = [client.post(path, **kwargs).status_code for _ in range(101)]
                self.assertEqual(set(statuses[:100]), {422}, (path, kwargs))
                self.assertEqual(statuses[100], 429, (path, kwargs))

    def test_valid_batch_pays_one_up_front_plus_the_rest(self):
        main._LIMITER = TokenBucketLimiter(
            RateLimitSettings(tickets_per_minute=60, burst=100), clock=lambda: 0.0
        )
        client = self.client("10.0.0.8")
        batch = {"tickets": [{"subject": "a"}] * 100}
        self.assertEqual(client.post("/classify-batch", json=batch).status_code, 200)
        self.assertEqual(main._LIMITER._buckets["10.0.0.8"][0], 0)
        self.assertEqual(post_one(client).status_code, 429)

    def test_tokens_refill_over_time(self):
        now = [0.0]
        limiter = TokenBucketLimiter(
            RateLimitSettings(tickets_per_minute=60, burst=100), clock=lambda: now[0]
        )
        self.assertEqual(limiter.check("a", 100), 0)
        self.assertGreater(limiter.check("a", 1), 0)
        now[0] = 5.0
        self.assertEqual(limiter.check("a", 5), 0)
        self.assertGreater(limiter.check("a", 1), 0)

    def test_disabled_limiter_allows_everything(self):
        limiter = TokenBucketLimiter(RateLimitSettings(enabled=False, burst=100))
        self.assertEqual(limiter.check("a", 10_000), 0)

    def test_client_table_is_bounded(self):
        limiter = TokenBucketLimiter(RateLimitSettings(max_clients=3))
        for index in range(10):
            limiter.check(f"c{index}", 1)
        self.assertEqual(len(limiter._buckets), 3)

    def test_settings_validation(self):
        self.assertEqual(rate_limit_from_config({}), RateLimitSettings())
        for bad in ({"burst": 10}, {"tickets_per_minute": 0}, {"enabled": 1},
                    {"nope": 1}):
            with self.assertRaises(ValueError):
                rate_limit_from_config({"rate_limit": bad})
        with self.assertRaises(ValueError):
            provider_queue_from_config({"provider_queue": {"max_depth": -1}})


class ClassifyTokenTests(unittest.TestCase):
    def setUp(self):
        for item in (
            patch("app.main.generate_classification", return_value=GOOD),
            patch("app.main.log_classification"),
            # Never queue here: a leaked slot must fail fast, not wait 30s per call.
            patch.object(main, "_QUEUE_SETTINGS", ProviderQueueSettings(0, 0.0)),
        ):
            item.start()
            self.addCleanup(item.stop)
        self.client = TestClient(app, client=("10.9.9.9", 1))

    def test_off_by_default(self):
        with patch.dict(os.environ, {}):
            os.environ.pop(main.CLASSIFY_TOKEN_ENV, None)
            self.assertEqual(post_one(self.client).status_code, 200)

    def test_enforced_when_configured(self):
        token = "s" * 32
        with patch.dict(os.environ, {main.CLASSIFY_TOKEN_ENV: token}):
            self.assertEqual(post_one(self.client).status_code, 401)
            self.assertEqual(
                post_one(self.client, headers={"X-Classify-Token": "x" * 32}).status_code,
                401,
            )
            self.assertEqual(
                post_one(self.client, headers={"X-Classify-Token": token}).status_code,
                200,
            )
            batch = {"tickets": [{"subject": "a"}]}
            self.assertEqual(
                self.client.post("/classify-batch", json=batch).status_code, 401
            )
            self.assertEqual(
                self.client.post(
                    "/classify-batch", json=batch,
                    headers={"X-Classify-Token": token},
                ).status_code,
                200,
            )

    def test_empty_or_blank_token_fails_closed(self):
        for value in ("", "   ", " " * 20):
            with self.subTest(value=value), patch.dict(
                os.environ, {main.CLASSIFY_TOKEN_ENV: value}
            ):
                self.assertTrue(main.classify_token_misconfigured())
                self.assertEqual(post_one(self.client).status_code, 503)
                self.assertEqual(
                    self.client.post(
                        "/classify-batch", json={"tickets": [{"subject": "a"}]}
                    ).status_code,
                    503,
                )

    def test_startup_logs_error_for_empty_token(self):
        async def run():
            async with main.lifespan(main.app):
                pass

        with patch.dict(os.environ, {main.CLASSIFY_TOKEN_ENV: ""}), patch(
            "app.main.startup_prune"
        ), self.assertLogs("app.main", level="ERROR"):
            asyncio.run(run())

    def test_weak_configured_token_fails_closed(self):
        with patch.dict(os.environ, {main.CLASSIFY_TOKEN_ENV: "short"}):
            self.assertEqual(post_one(self.client).status_code, 503)


class ProviderGateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.held = [ProviderCall() for _ in range(MAX_CONCURRENT_PROVIDER_CALLS)]
        self.assertTrue(all(call.admitted for call in self.held))

    async def queued(self, count):
        """Yield to the loop until exactly `count` requests are queued."""
        while queued_provider_waiters() != count:
            await asyncio.sleep(0)

    async def asyncTearDown(self):
        for call in self.held:
            call.cancel()
        # all slots and queue positions must be free again
        self.assertEqual(queued_provider_waiters(), 0)
        probes = [ProviderCall() for _ in range(MAX_CONCURRENT_PROVIDER_CALLS)]
        self.assertTrue(all(call.admitted for call in probes))
        for call in probes:
            call.cancel()

    @bounded
    async def test_waiters_are_served_in_order_when_slots_free(self):
        order = []

        async def waiter(name):
            call, reason = await acquire_provider_call(8, 5.0)
            order.append((name, reason))
            return call

        tasks = []
        for position, name in enumerate(("first", "second", "third"), start=1):
            tasks.append(asyncio.create_task(waiter(name)))
            await self.queued(position)
        self.assertEqual(queued_provider_waiters(), 3)
        # A newcomer must not overtake the queue when a slot frees.
        self.held.pop().cancel()
        first = await asyncio.wait_for(tasks[0], 1)
        self.assertTrue(first.admitted)
        late = ProviderCall()
        self.assertFalse(late.admitted)
        self.held.pop().cancel()
        second = await asyncio.wait_for(tasks[1], 1)
        first.cancel()
        third = await asyncio.wait_for(tasks[2], 1)
        self.assertEqual([n for n, _ in order], ["first", "second", "third"])
        second.cancel()
        third.cancel()

    @bounded
    async def test_queue_full_returns_marked_fallback_through_endpoint(self):
        waiting = asyncio.create_task(acquire_provider_call(1, 5.0))
        await self.queued(1)
        call, reason = await acquire_provider_call(1, 5.0)
        self.assertIsNone(call)
        self.assertEqual(reason, "queue_full")
        with (
            patch.object(main, "_QUEUE_SETTINGS", ProviderQueueSettings(1, 5.0)),
            patch("app.main.log_classification"),
        ):
            result = await main.classify(main.Ticket(subject="Payment failed"))
        self.assertEqual(result.category, "Other / Needs Review")
        self.assertTrue(result.human_review)
        self.assertTrue(result.reason.startswith("provider_busy"))
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting

    @bounded
    async def test_wait_expiry_returns_marked_fallback_and_frees_position(self):
        with (
            patch.object(main, "_QUEUE_SETTINGS", ProviderQueueSettings(4, 0.05)),
            patch("app.main.log_classification"),
        ):
            result = await main.classify(main.Ticket(subject="Payment failed"))
        self.assertEqual(result.reason, "provider_busy")
        self.assertEqual(queued_provider_waiters(), 0)

    @bounded
    async def test_busy_marker_survives_rule_overrides(self):
        ticket = main.Ticket(subject="my account was hacked")
        with (
            patch.object(main, "_QUEUE_SETTINGS", ProviderQueueSettings(0, 0.0)),
            patch("app.main.log_classification"),
        ):
            result = await main.classify(ticket)
        self.assertTrue(result.reason.startswith("provider_busy"))
        self.assertTrue(result.human_review)

    @bounded
    async def test_busy_batch_items_count_as_degraded(self):
        request = main.BatchClassificationRequest(tickets=[main.Ticket(subject="a")])
        with (
            patch.object(main, "_QUEUE_SETTINGS", ProviderQueueSettings(0, 0.0)),
            patch("app.main.log_classification"),
        ):
            response = await main.classify_batch(request)
        self.assertEqual(response.degraded, 1)
        self.assertTrue(response.results[0].reason.startswith("provider_busy"))

    @bounded
    async def test_cancellation_while_queued_releases_queue_slot(self):
        task = asyncio.create_task(acquire_provider_call(1, 5.0))
        await self.queued(1)
        self.assertEqual(queued_provider_waiters(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(queued_provider_waiters(), 0)
        # depth 1 is free again
        again = asyncio.create_task(acquire_provider_call(1, 5.0))
        await self.queued(1)
        self.assertEqual(queued_provider_waiters(), 1)
        again.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await again

    @bounded
    async def test_cancel_after_handoff_passes_the_slot_on(self):
        first = asyncio.create_task(acquire_provider_call(4, 5.0))
        second = asyncio.create_task(acquire_provider_call(4, 5.0))
        await self.queued(2)
        self.held.pop().cancel()  # slot handed to `first`
        first.cancel()  # cancelled before it resumes
        call, _ = await asyncio.wait_for(second, 1)
        self.assertTrue(call.admitted)
        call.cancel()
        try:
            await first
        except asyncio.CancelledError:
            pass
        else:  # it won the race and holds the slot; release it
            (await first)[0].cancel()


if __name__ == "__main__":
    unittest.main()
