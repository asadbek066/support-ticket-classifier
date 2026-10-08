import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import audit, main

TOKEN = "t" * 32

BASE = """
categories:
  - Billing & Payments
  - Other / Needs Review
queue_map:
  Billing & Payments: billing-queue
  Other / Needs Review: triage
rules:
  confidence_threshold: 0.65
ollama:
  model: model-a
  api_url: http://localhost:11434/api/generate
audit:
  max_text_chars: 111
"""
CHANGED = (
    BASE.replace("model-a", "model-b")
    .replace("localhost:11434", "localhost:9999")
    .replace("Billing & Payments", "Payments")
    .replace("max_text_chars: 111", "max_text_chars: 222")
)


class ConfigReloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "config.yaml"
        self.path.write_text(BASE, encoding="utf-8")
        for item in (
            patch.object(main, "CONFIG_PATH", self.path),
            patch.object(main, "_ACTIVE", main._ACTIVE),
            patch.object(main, "_LIMITER", main._LIMITER),
            patch.object(main, "_QUEUE_SETTINGS", main._QUEUE_SETTINGS),
            patch.object(main.rules, "cfg", main.rules.cfg),
            patch.object(audit, "_SETTINGS", audit.get_settings()),
            patch.dict(os.environ, {"TICKET_CLASSIFIER_ADMIN_TOKEN": TOKEN}),
        ):
            item.start()
            self.addCleanup(item.stop)
        main.reload_active_config()
        self.client = TestClient(main.app)

    def reload(self):
        return self.client.post("/reload-config", headers={"X-Admin-Token": TOKEN})

    def state(self):
        config = main.load_config()
        categories, model, url = main.runtime_config(config)
        return (
            model,
            url,
            main.rules.configured_categories(),
            categories,
            audit.get_settings().max_text_chars,
        )

    def test_edit_applies_nothing_until_reload(self):
        before = self.state()
        self.path.write_text(CHANGED, encoding="utf-8")
        self.assertEqual(self.state(), before)

    def test_valid_reload_applies_model_url_rules_and_audit_together(self):
        self.path.write_text(CHANGED, encoding="utf-8")
        self.assertEqual(self.reload().status_code, 200)
        model, url, rule_categories, categories, text_chars = self.state()
        self.assertEqual(model, "model-b")
        self.assertEqual(url, "http://localhost:9999/api/generate")
        self.assertEqual(rule_categories, ["Payments", "Other / Needs Review"])
        self.assertEqual(categories, rule_categories)
        self.assertEqual(text_chars, 222)

    def test_invalid_reload_keeps_previous_model_url_rules_and_audit(self):
        before = self.state()
        bad_configs = (
            ":\n  broken: [",
            "- a\n- list\n",
            # the rules section is fine, the new model is not
            CHANGED.replace("model-b", "  "),
            # the model/url/rules are fine, the audit section is not
            CHANGED.replace("max_text_chars: 222", "max_text_chars: -5"),
            CHANGED.replace("max_text_chars: 222", "max_text_chars: 5\n  bogus: 1"),
            CHANGED + "\nrate_limit:\n  burst: 1\n",
            CHANGED.replace("categories:", "categories: nope\nold:"),
            CHANGED.replace("localhost:9999", "localhost:9999").replace(
                "http://", "ftp://"
            ),
        )
        for text in bad_configs:
            with self.subTest(text=text[:40]):
                self.path.write_text(text, encoding="utf-8")
                self.assertEqual(self.reload().status_code, 503)
                self.assertEqual(self.state(), before)

    def test_missing_file_keeps_previous_config(self):
        before = self.state()
        self.path.unlink()
        self.assertEqual(self.reload().status_code, 503)
        self.assertEqual(self.state(), before)

    def test_requests_use_the_active_generation_not_the_file(self):
        self.path.write_text(CHANGED, encoding="utf-8")  # edited, not reloaded
        seen = {}

        def fake_generate(ticket, categories, model, api_url, request_control=None):
            seen.update(model=model, api_url=api_url, categories=categories)
            return {"category": "Billing & Payments", "confidence": 0.9,
                    "human_review": False}

        with (
            patch("app.main.generate_classification", fake_generate),
            patch("app.main.log_classification"),
        ):
            response = self.client.post("/classify", json={"subject": "x"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(seen["model"], "model-a")
        self.assertEqual(seen["api_url"], "http://localhost:11434/api/generate")
        self.assertEqual(seen["categories"], ["Billing & Payments", "Other / Needs Review"])

    def test_rate_limiter_state_survives_reload_when_settings_unchanged(self):
        limiter = main._LIMITER
        self.assertEqual(self.reload().status_code, 200)
        self.assertIs(main._LIMITER, limiter)


if __name__ == "__main__":
    unittest.main()
