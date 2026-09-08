import unittest
from unittest.mock import patch

from app.ollama_client import FALLBACK_REASON, generate_classification


class OllamaBoundaryTests(unittest.TestCase):
    @patch("app.ollama_client.httpx.post", side_effect=RuntimeError("provider secret response"))
    def test_provider_errors_do_not_become_audit_reason_text(self, _post):
        result = generate_classification({}, ["Billing"], "model", "http://127.0.0.1:11434/api/generate")

        self.assertEqual(result["reason"], FALLBACK_REASON)
        self.assertNotIn("provider secret response", result["reason"])
        self.assertTrue(result["human_review"])


if __name__ == "__main__":
    unittest.main()
