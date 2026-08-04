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

        with self.assertRaisesRegex(ValueError, "mapping"):
            self.engine.reload()

        self.assertIs(self.engine.cfg, original)
        result = self.engine.apply({}, {"category": "Billing", "confidence": 0.9})
        self.assertEqual(result["queue"], "billing")


if __name__ == "__main__":
    unittest.main()
