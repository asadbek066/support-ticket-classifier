import os
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app, classify_batch, runtime_config, safe_api_url
from app.schemas import BatchClassificationRequest, Ticket


class MainBoundaryTests(unittest.TestCase):
    def test_admin_surface_fails_closed_without_configured_token(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TICKET_CLASSIFIER_ADMIN_TOKEN", None)
            response = TestClient(app).get("/audit-logs")

        self.assertEqual(response.status_code, 503)

    def test_admin_surface_requires_the_configured_token(self):
        token = "t" * 32
        with patch.dict(os.environ, {"TICKET_CLASSIFIER_ADMIN_TOKEN": token}):
            client = TestClient(app)
            self.assertEqual(client.get("/audit-logs").status_code, 401)
            self.assertEqual(
                client.get(
                    "/audit-logs", headers={"X-Admin-Token": "wrong"}
                ).status_code,
                401,
            )
            with patch("app.main.read_audit_logs", return_value=[]):
                response = client.get("/audit-logs", headers={"X-Admin-Token": token})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"total": 0, "logs": []})

    def test_audit_response_reports_total_beyond_requested_page(self):
        token = "t" * 32
        with (
            patch.dict(os.environ, {"TICKET_CLASSIFIER_ADMIN_TOKEN": token}),
            patch("app.main.read_audit_logs", return_value=[]),
            patch("app.main.count_audit_logs", return_value=123),
        ):
            response = TestClient(app).get(
                "/audit-logs?limit=5",
                headers={"X-Admin-Token": token},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total"], 123)

    def test_status_url_removes_credentials_and_query_data(self):
        self.assertEqual(
            safe_api_url(
                "http://user:secret@example.test:11434/api/generate?token=private"
            ),
            "http://example.test:11434/api/generate",
        )

    def test_runtime_config_rejects_non_string_provider_settings(self):
        with self.assertRaisesRegex(RuntimeError, "configuration unavailable"):
            runtime_config({"ollama": {"model": ["not-a-model"]}})


class BatchBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_deadline_returns_review_fallbacks_for_remaining_tickets(self):
        request = BatchClassificationRequest(
            tickets=[Ticket(subject="first"), Ticket(subject="second")]
        )

        def slow_provider(*_args):
            time.sleep(0.05)
            return {"category": "Billing", "confidence": 0.9}

        with (
            patch("app.main.BATCH_TIMEOUT_SECONDS", 0.01),
            patch("app.main.generate_classification", side_effect=slow_provider),
        ):
            response = await classify_batch(request)

        self.assertEqual(response.total, 2)
        self.assertEqual(len(response.results), 2)
        self.assertTrue(all(result.human_review for result in response.results))
        self.assertTrue(
            all(
                result.reason == "classification unavailable"
                for result in response.results
            )
        )


if __name__ == "__main__":
    unittest.main()
