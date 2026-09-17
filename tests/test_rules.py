import tempfile
import unittest
from pathlib import Path

from app.rules import RulesEngine

VALID_CONFIG = """
rules:
  confidence_threshold: 0.65
queue_map:
  Billing: billing
"""


class RulesEngineReloadTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "config.yaml"
        self.config_path.write_text(VALID_CONFIG, encoding="utf-8")
        self.engine = RulesEngine(str(self.config_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_empty_config_is_rejected_without_losing_active_rules(self):
        original = self.engine.cfg
        self.config_path.write_text("", encoding="utf-8")

        with self.assertRaisesRegex((TypeError, ValueError), "mapping"):
            self.engine.reload()

        self.assertIs(self.engine.cfg, original)
        result = self.engine.apply({}, {"category": "Billing", "confidence": 0.9})
        self.assertEqual(result["queue"], "billing")

    def test_model_output_is_normalized_to_configured_categories_and_queues(self):
        self.config_path.write_text(
            """
categories:
  - Billing
queue_map:
  Billing: billing
rules:
  confidence_threshold: 0.65
""",
            encoding="utf-8",
        )
        engine = RulesEngine(str(self.config_path))

        result = engine.apply(
            {"customer_type": "enterprise"},
            {
                "category": "<script>alert(1)</script>",
                "confidence": float("nan"),
                "queue": "attacker-controlled-queue",
                "reason": "x" * 3000,
                "human_review": "false",
            },
        )

        self.assertEqual(result["category"], "Other / Needs Review")
        self.assertEqual(result["queue"], "triage")
        self.assertEqual(result["confidence"], 0.0)
        self.assertTrue(result["human_review"])
        self.assertEqual(len(result["reason"]), 2000)

    def test_configured_queue_wins_over_model_queue(self):
        result = self.engine.apply(
            {},
            {"category": "Billing", "confidence": 0.9, "queue": "wrong-queue"},
        )
        self.assertEqual(result["queue"], "billing")


class RulesDecisionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "config.yaml"
        self.config_path.write_text(
            """
categories:
  - Billing
  - Security Concerns
queue_map:
  Billing: billing
  Security Concerns: security-ops
rules:
  confidence_threshold: 0.65
  enterprise_confidence_boost: 0.1
  forced_human_review_categories:
    - Security Concerns
""",
            encoding="utf-8",
        )
        self.engine = RulesEngine(str(self.config_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def _apply(self, ticket, model_out):
        return self.engine.apply(ticket, model_out)

    def test_configured_categories_are_returned(self):
        self.assertEqual(
            self.engine.configured_categories(), ["Billing", "Security Concerns"]
        )

    def test_enterprise_boost_can_lift_a_result_above_the_threshold(self):
        result = self._apply(
            {"customer_type": "enterprise"},
            {
                "category": "Billing",
                "confidence": 0.6,
                "human_review": False,
            },
        )

        self.assertAlmostEqual(result["confidence"], 0.7)
        self.assertFalse(result["human_review"])

    def test_below_threshold_without_boost_requires_review(self):
        result = self._apply(
            {"customer_type": "free"},
            {"category": "Billing", "confidence": 0.6, "human_review": False},
        )

        self.assertAlmostEqual(result["confidence"], 0.6)
        self.assertTrue(result["human_review"])

    def test_exact_case_boost_only_applies_to_enterprise(self):
        result = self._apply(
            {"customer_type": "ENTERPRISE"},
            {"category": "Billing", "confidence": 0.6, "human_review": False},
        )
        self.assertFalse(result["human_review"])

        result = self._apply(
            {"customer_type": "enterprise-ish"},
            {"category": "Billing", "confidence": 0.6, "human_review": False},
        )
        self.assertTrue(result["human_review"])

    def test_forced_category_always_requires_review(self):
        result = self._apply(
            {},
            {"category": "Security Concerns", "confidence": 0.99, "human_review": False},
        )

        self.assertEqual(result["category"], "Security Concerns")
        self.assertEqual(result["queue"], "security-ops")
        self.assertTrue(result["human_review"])

    def test_model_requested_review_is_preserved(self):
        result = self._apply(
            {},
            {"category": "Billing", "confidence": 0.99, "human_review": True},
        )

        self.assertTrue(result["human_review"])

    def test_invalid_threshold_falls_back_to_default(self):
        self.config_path.write_text(
            """
categories:
  - Billing
queue_map:
  Billing: billing
rules:
  confidence_threshold: not-a-number
""",
            encoding="utf-8",
        )
        engine = RulesEngine(str(self.config_path))

        result = engine.apply(
            {},
            {"category": "Billing", "confidence": 0.7, "human_review": False},
        )

        self.assertFalse(result["human_review"])

    def test_apply_uses_the_provided_snapshot_after_a_reload(self):
        snapshot = self.engine.snapshot()
        categories = self.engine.configured_categories(snapshot)

        self.config_path.write_text(
            """
categories:
  - Refunds
queue_map:
  Refunds: refunds
rules:
  confidence_threshold: 0.65
""",
            encoding="utf-8",
        )
        self.engine.reload()

        result = self.engine.apply(
            {}, {"category": "Billing", "confidence": 0.9}, snapshot
        )

        self.assertEqual(categories, ["Billing", "Security Concerns"])
        self.assertEqual(result["category"], "Billing")
        self.assertEqual(result["queue"], "billing")


if __name__ == "__main__":
    unittest.main()
