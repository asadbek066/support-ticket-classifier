import unittest
from typing import ClassVar
from unittest.mock import patch

from app.ollama_client import (
    FALLBACK_REASON,
    MAX_RESPONSE_CHARS,
    generate_classification,
)


class OllamaBoundaryTests(unittest.TestCase):
    @patch(
        "app.ollama_client.httpx.stream",
        side_effect=RuntimeError("provider secret response"),
    )
    def test_provider_errors_do_not_become_audit_reason_text(self, _post):
        result = generate_classification(
            {}, ["Billing"], "model", "http://127.0.0.1:11434/api/generate"
        )

        self.assertEqual(result["reason"], FALLBACK_REASON)
        self.assertNotIn("provider secret response", result["reason"])
        self.assertTrue(result["human_review"])

    def test_oversized_provider_body_is_rejected_while_streaming(self):
        class OversizedStream:
            headers: ClassVar[dict] = {}
            encoding = "utf-8"

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def raise_for_status(self):
                return None

            def iter_bytes(self):
                yield b"x" * (MAX_RESPONSE_CHARS + 1)

        with patch(
            "app.ollama_client.httpx.stream", return_value=OversizedStream()
        ) as stream:
            result = generate_classification(
                {}, ["Billing"], "model", "http://127.0.0.1:11434/api/generate"
            )

        stream.assert_called_once()
        self.assertEqual(result["reason"], FALLBACK_REASON)


if __name__ == "__main__":
    unittest.main()
