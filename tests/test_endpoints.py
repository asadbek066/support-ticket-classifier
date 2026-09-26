import asyncio
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from app.main import app
from app.schemas import Ticket

TOKEN = "t" * 32
ADMIN_ENV = {"TICKET_CLASSIFIER_ADMIN_TOKEN": TOKEN}


class ClassifyEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_classify_returns_routed_result_and_audits(self):
        with (
            patch(
                "app.main.generate_classification",
                return_value={
                    "category": "Billing & Payments",
                    "confidence": 0.9,
                    "reason": "payment keywords",
                    "human_review": False,
                },
            ),
            patch("app.main.log_classification") as audit,
        ):
            response = self.client.post(
                "/classify", json={"subject": "Payment failed"}
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["category"], "Billing & Payments")
        self.assertEqual(body["queue"], "billing-queue")
        self.assertFalse(body["human_review"])
        audit.assert_called_once()

    def test_classify_provider_failure_returns_review_fallback(self):
        with (
            patch(
                "app.main.generate_classification",
                return_value=main.provider_fallback(),
            ),
            patch("app.main.log_classification") as audit,
        ):
            response = self.client.post("/classify", json={"subject": "x"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["category"], "Other / Needs Review")
        self.assertEqual(body["queue"], "triage")
        self.assertTrue(body["human_review"])
        self.assertEqual(body["reason"], "model-unreachable-or-invalid-response")
        audit.assert_called_once()

    def test_classify_timeout_returns_review_fallback(self):
        def slow_provider(*_args, **_kwargs):
            time.sleep(0.05)
            return {"category": "Billing & Payments", "confidence": 0.9}

        with (
            patch("app.main.CLASSIFY_TIMEOUT_SECONDS", 0.01),
            patch("app.main.generate_classification", side_effect=slow_provider),
            patch("app.main.log_classification"),
        ):
            response = self.client.post("/classify", json={"subject": "x"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["human_review"])
        self.assertEqual(
            response.json()["reason"], "model-unreachable-or-invalid-response"
        )

    def test_classify_config_failure_is_503(self):
        with patch("app.main.load_config", side_effect=RuntimeError("bad config")):
            response = self.client.post("/classify", json={"subject": "x"})

        self.assertEqual(response.status_code, 503)

    def test_classify_audit_failure_does_not_discard_result(self):
        with (
            patch(
                "app.main.generate_classification",
                return_value={
                    "category": "Billing & Payments",
                    "confidence": 0.9,
                    "human_review": False,
                },
            ),
            patch("app.main.log_classification", side_effect=OSError("disk full")),
        ):
            response = self.client.post("/classify", json={"subject": "x"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["category"], "Billing & Payments")

    def test_classify_unknown_field_is_422(self):
        response = self.client.post(
            "/classify", json={"subject": "x", "attacker_controlled": True}
        )
        self.assertEqual(response.status_code, 422)

    def test_batch_request_shape_is_validated(self):
        self.assertEqual(
            self.client.post("/classify-batch", json={"tickets": []}).status_code,
            422,
        )
        self.assertEqual(
            self.client.post(
                "/classify-batch", json={"tickets": [{"subject": "x"}] * 101}
            ).status_code,
            422,
        )

    def test_request_body_limit_returns_413_with_security_headers(self):
        oversized = b"x" * (main.MAX_REQUEST_BODY_BYTES + 1)
        response = self.client.post(
            "/classify",
            content=oversized,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")

class AuditBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_classify_audit_timeout_does_not_wait_for_the_write(self):
        release = threading.Event()

        def blocking_audit(*_args):
            release.wait(5)

        with (
            patch(
                "app.main.generate_classification",
                return_value={
                    "category": "Billing & Payments",
                    "confidence": 0.9,
                    "human_review": False,
                },
            ),
            patch("app.main.log_classification", side_effect=blocking_audit),
            patch("app.main.AUDIT_WRITE_TIMEOUT_SECONDS", 0.05),
        ):
            started = time.monotonic()
            response = await main.classify(Ticket(subject="x"))
            elapsed = time.monotonic() - started
            release.set()

        self.assertEqual(response.category, "Billing & Payments")
        self.assertLess(elapsed, 1.0)

    async def test_stalled_audit_writer_does_not_occupy_inference_executor(self):
        from concurrent.futures import ThreadPoolExecutor

        release = threading.Event()
        started = threading.Event()
        finished = threading.Event()
        loop = asyncio.get_running_loop()
        inference_executor = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(inference_executor)

        def blocking_audit(*_args):
            started.set()
            release.wait(5.0)
            finished.set()

        try:
            with (
                patch("app.main.log_classification", side_effect=blocking_audit),
                patch("app.main.AUDIT_WRITE_TIMEOUT_SECONDS", 0.02),
            ):
                await main._audit_classification({}, {})
                self.assertTrue(started.wait(1.0))
                result = await asyncio.wait_for(
                    asyncio.to_thread(lambda: "inference executor is available"),
                    timeout=0.1,
                )
                self.assertEqual(result, "inference executor is available")
        finally:
            release.set()
            await asyncio.to_thread(finished.wait, 1.0)
            inference_executor.shutdown(wait=True)

class ProviderTimeoutBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_classify_timeout_closes_slow_header_connection(self):
        import socketserver
        import time

        from app.ollama_client import MAX_CONCURRENT_PROVIDER_CALLS, ProviderCall

        started = threading.Event()
        server_finished = threading.Event()
        stop_stream = threading.Event()

        class SlowHeaderHandler(socketserver.BaseRequestHandler):
            def handle(self):
                try:
                    self.request.recv(64 * 1024)
                    self.request.sendall(b"HTTP/1.1 200 OK\r\nX-Slow: ")
                    started.set()
                    for _ in range(100):
                        if stop_stream.wait(0.03):
                            break
                        self.request.sendall(b"a")
                    self.request.sendall(
                        b"\r\nContent-Type: application/json\r\n"
                        b"Content-Length: 2\r\n\r\n{}"
                    )
                except OSError:
                    pass
                finally:
                    server_finished.set()

        server = socketserver.ThreadingTCPServer(
            ("127.0.0.1", 0), SlowHeaderHandler
        )
        server.daemon_threads = True
        server.block_on_close = False
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        url = f"http://127.0.0.1:{server.server_address[1]}/api/generate"

        try:
            with (
                patch("app.main._provider_settings", return_value=("model", url)),
                patch("app.main.CLASSIFY_TIMEOUT_SECONDS", 0.25),
                patch("app.main.log_classification"),
            ):
                result = await main.classify(Ticket(subject="Payment failed"))

            self.assertEqual(result.category, "Other / Needs Review")
            self.assertTrue(result.human_review)
            self.assertTrue(started.wait(1.0))
            self.assertTrue(server_finished.wait(1.0))

            deadline = time.monotonic() + 1.0
            admitted_count = 0
            while time.monotonic() < deadline:
                controls = [
                    ProviderCall()
                    for _ in range(MAX_CONCURRENT_PROVIDER_CALLS + 1)
                ]
                admitted_count = sum(control.admitted for control in controls)
                for control in controls:
                    control.cancel()
                if admitted_count == MAX_CONCURRENT_PROVIDER_CALLS:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(admitted_count, MAX_CONCURRENT_PROVIDER_CALLS)
        finally:
            stop_stream.set()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=1.0)


class HealthAndHeadersTests(unittest.TestCase):
    def test_health_needs_no_admin_token(self):
        response = TestClient(app).get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_security_headers_are_present(self):
        response = TestClient(app).get("/health")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertIn("default-src", response.headers["content-security-policy"])
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_dashboard_html_keeps_default_cache_headers(self):
        response = TestClient(app).get("/")
        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(response.headers.get("cache-control"), "no-store")

    def test_generated_docs_are_exempt_from_csp(self):
        response = TestClient(app).get("/docs")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("content-security-policy", response.headers)
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")


class VerifyOllamaEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_requires_admin_token(self):
        with patch.dict(os.environ, ADMIN_ENV, clear=False):
            self.assertEqual(
                self.client.get("/verify-ollama-connection").status_code, 401
            )

    def test_reports_connected_with_missing_model(self):
        with (
            patch.dict(os.environ, ADMIN_ENV, clear=False),
            patch(
                "app.main.check_ollama_connection",
                return_value={"connected": True, "model_available": False},
            ),
        ):
            response = self.client.get(
                "/verify-ollama-connection", headers={"X-Admin-Token": TOKEN}
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["connected"])
        self.assertFalse(body["model_available"])
        self.assertEqual(body["api_url"], "http://localhost:11434/api/generate")
        self.assertEqual(body["model"], "deepseek-r1:1.5b")

    def test_config_failure_is_503(self):
        with (
            patch.dict(os.environ, ADMIN_ENV, clear=False),
            patch("app.main.load_config", side_effect=RuntimeError("bad")),
        ):
            response = self.client.get(
                "/verify-ollama-connection", headers={"X-Admin-Token": TOKEN}
            )

        self.assertEqual(response.status_code, 503)


class ReloadConfigEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_requires_admin_token(self):
        with patch.dict(os.environ, ADMIN_ENV, clear=False):
            self.assertEqual(self.client.post("/reload-config").status_code, 401)

    def test_reload_succeeds_with_token(self):
        with patch.dict(os.environ, ADMIN_ENV, clear=False):
            response = self.client.post(
                "/reload-config", headers={"X-Admin-Token": TOKEN}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_reload_failure_keeps_previous_rules(self):
        with (
            patch.dict(os.environ, ADMIN_ENV, clear=False),
            patch.object(main.rules, "reload", side_effect=OSError("io error")),
        ):
            response = self.client.post(
                "/reload-config", headers={"X-Admin-Token": TOKEN}
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("Billing & Payments", main.rules.configured_categories())


class LoadConfigCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "config.yaml"
        self.config_path.write_text("categories:\n  - Billing\n", encoding="utf-8")
        self.path_patch = patch.object(main, "CONFIG_PATH", self.config_path)
        self.cache_patch = patch.dict(main._CONFIG_CACHE, {}, clear=True)
        self.path_patch.start()
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)
        self.addCleanup(self.path_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)

    def test_returns_parsed_mapping(self):
        self.assertEqual(main.load_config(), {"categories": ["Billing"]})

    def test_cached_value_is_reused_until_the_file_changes(self):
        first = main.load_config()
        self.assertIs(main.load_config(), first)

        self.config_path.write_text(
            "categories:\n  - Refunds\n  - Billing\n", encoding="utf-8"
        )
        self.assertEqual(
            main.load_config(), {"categories": ["Refunds", "Billing"]}
        )

    def test_malformed_config_falls_back_to_last_known_good(self):
        good = main.load_config()
        self.config_path.write_text(":\n  broken: [", encoding="utf-8")

        self.assertEqual(main.load_config(), good)

    def test_missing_file_falls_back_to_last_known_good(self):
        good = main.load_config()
        self.config_path.unlink()

        self.assertEqual(main.load_config(), good)

    def test_missing_file_raises_without_a_cached_value(self):
        self.config_path.unlink()
        with self.assertRaisesRegex(RuntimeError, "configuration unavailable"):
            main.load_config()

    def test_non_mapping_config_raises(self):
        self.config_path.write_text("- just\n- a list\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "configuration unavailable"):
            main.load_config()


if __name__ == "__main__":
    unittest.main()
