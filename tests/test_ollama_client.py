import json
import unittest
from typing import ClassVar
from unittest.mock import patch

import httpx

from app.ollama_client import (
    FALLBACK_REASON,
    MAX_RESPONSE_CHARS,
    check_ollama_connection,
    generate_classification,
    is_provider_fallback,
    provider_fallback,
)

API_URL = "http://127.0.0.1:11434/api/generate"


class StubStream:
    def __init__(self, chunks, headers=None, encoding="utf-8", error=None):
        self.chunks = chunks
        self.headers = headers or {}
        self.encoding = encoding
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def iter_bytes(self):
        yield from self.chunks


def classify_stream(chunks, headers=None, encoding="utf-8"):
    with patch(
        "app.ollama_client.httpx.stream",
        return_value=StubStream(chunks, headers=headers, encoding=encoding),
    ):
        return generate_classification({}, ["Billing"], "model", API_URL)


class OllamaBoundaryTests(unittest.TestCase):
    @patch(
        "app.ollama_client.httpx.stream",
        side_effect=RuntimeError("provider secret response"),
    )
    def test_provider_errors_do_not_become_audit_reason_text(self, _post):
        result = generate_classification({}, ["Billing"], "model", API_URL)

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
            result = generate_classification({}, ["Billing"], "model", API_URL)

        stream.assert_called_once()
        self.assertEqual(result["reason"], FALLBACK_REASON)

    def test_content_length_lie_is_capped_by_streaming_guard(self):
        payload = json.dumps({"response": "{" * (MAX_RESPONSE_CHARS + 1)}).encode()

        result = classify_stream([payload], headers={"content-length": "10"})

        self.assertEqual(result["reason"], FALLBACK_REASON)


class OllamaParsingTests(unittest.TestCase):
    def test_wrapped_response_is_parsed(self):
        inner = {"category": "Billing", "confidence": 0.9, "human_review": False}
        payload = json.dumps({"response": json.dumps(inner)}).encode()

        result = classify_stream([payload])

        self.assertEqual(result, inner)

    def test_text_field_is_parsed(self):
        inner = {"category": "Billing", "confidence": 0.8}
        payload = json.dumps({"text": json.dumps(inner)}).encode()

        result = classify_stream([payload])

        self.assertEqual(result, inner)

    def test_plain_json_object_is_parsed(self):
        inner = {"category": "Billing", "confidence": 0.7}
        payload = json.dumps(inner).encode()

        result = classify_stream([payload])

        self.assertEqual(result, inner)

    def test_multichunk_stream_is_reassembled(self):
        inner = {"category": "Billing", "confidence": 0.95}
        raw = json.dumps({"response": json.dumps(inner)}).encode()
        midpoint = len(raw) // 2

        result = classify_stream([raw[:midpoint], raw[midpoint:]])

        self.assertEqual(result, inner)

    def test_done_line_wins_in_multi_line_stream(self):
        inner = {"category": "Billing", "confidence": 0.85}
        lines = [
            json.dumps({"response": "thinking out loud"}),
            json.dumps({"response": "still thinking"}),
            json.dumps({"response": json.dumps(inner), "done": True}),
            "",
        ]
        payload = "\n".join(lines).encode()

        result = classify_stream([payload])

        self.assertEqual(result, inner)

    def test_prose_wrapped_json_is_extracted(self):
        inner = {"category": "Billing", "confidence": 0.75}
        text = f"Here you go: {json.dumps(inner)} — hope that helps"
        payload = json.dumps({"response": text}).encode()

        result = classify_stream([payload])

        self.assertEqual(result, inner)

    def test_unterminated_brace_run_degrades_to_fallback(self):
        text = "{" * (MAX_RESPONSE_CHARS - 100)
        payload = json.dumps({"response": text}).encode()

        result = classify_stream([payload])

        self.assertEqual(result["reason"], FALLBACK_REASON)

    def test_invalid_utf8_degrades_to_fallback(self):
        result = classify_stream([b"\xff\xfe\x00"], encoding="utf-8")

        self.assertEqual(result["reason"], FALLBACK_REASON)

    def test_stream_deadline_is_enforced(self):
        payload = json.dumps({"response": "{}"}).encode()

        with patch("app.ollama_client.MAX_STREAM_SECONDS", 0.0):
            result = classify_stream([payload])

        self.assertEqual(result["reason"], FALLBACK_REASON)


def _tags_response(models):
    return httpx.Response(
        status_code=200,
        json={"models": models},
        request=httpx.Request("GET", "http://127.0.0.1:11434/api/tags"),
    )


class OllamaConnectionCheckTests(unittest.TestCase):
    def test_uses_the_tags_endpoint(self):
        with patch(
            "app.ollama_client.httpx.get", return_value=_tags_response([])
        ) as get:
            check_ollama_connection(API_URL, "deepseek-r1:1.5b")

        self.assertEqual(get.call_args.args[0], "http://127.0.0.1:11434/api/tags")

    def test_keeps_a_reverse_proxy_path_prefix(self):
        with patch(
            "app.ollama_client.httpx.get", return_value=_tags_response([])
        ) as get:
            check_ollama_connection(
                "http://proxy.test/ollama/api/generate", "deepseek-r1:1.5b"
            )

        self.assertEqual(
            get.call_args.args[0], "http://proxy.test/ollama/api/tags"
        )

    def test_connected_and_model_present(self):
        models = [{"name": "deepseek-r1:1.5b", "model": "deepseek-r1:1.5b"}]
        with patch(
            "app.ollama_client.httpx.get", return_value=_tags_response(models)
        ):
            result = check_ollama_connection(API_URL, "deepseek-r1:1.5b")

        self.assertEqual(
            result, {"connected": True, "model_available": True}
        )

    def test_untagged_model_matches_latest_alias(self):
        models = [{"name": "deepseek-r1:latest"}]
        with patch(
            "app.ollama_client.httpx.get", return_value=_tags_response(models)
        ):
            result = check_ollama_connection(API_URL, "deepseek-r1")

        self.assertTrue(result["model_available"])

    def test_connected_with_model_missing(self):
        models = [{"name": "llama3:latest"}]
        with patch(
            "app.ollama_client.httpx.get", return_value=_tags_response(models)
        ):
            result = check_ollama_connection(API_URL, "deepseek-r1:1.5b")

        self.assertTrue(result["connected"])
        self.assertFalse(result["model_available"])

    def test_server_error_is_not_connected(self):
        response = httpx.Response(
            status_code=500,
            request=httpx.Request("GET", "http://127.0.0.1:11434/api/tags"),
        )
        with patch("app.ollama_client.httpx.get", return_value=response):
            result = check_ollama_connection(API_URL, "deepseek-r1:1.5b")

        self.assertFalse(result["connected"])
        self.assertEqual(result["status_code"], 500)

    def test_transport_error_returns_hint(self):
        with patch(
            "app.ollama_client.httpx.get",
            side_effect=httpx.ConnectError("refused"),
        ):
            result = check_ollama_connection(API_URL, "deepseek-r1:1.5b")

        self.assertFalse(result["connected"])
        self.assertIn("hint", result)

    def test_invalid_url_degrades_without_raising(self):
        result = check_ollama_connection("http://:::", "deepseek-r1:1.5b")

        self.assertFalse(result["connected"])


class ProviderFallbackTests(unittest.TestCase):
    def test_fallback_shape_is_stable(self):
        self.assertEqual(
            provider_fallback(),
            {
                "category": "Other / Needs Review",
                "confidence": 0.0,
                "queue": "triage",
                "reason": FALLBACK_REASON,
                "human_review": True,
                "_provider_fallback": True,
            },
        )
        self.assertTrue(is_provider_fallback(provider_fallback()))
        self.assertFalse(
            is_provider_fallback(
                {"reason": FALLBACK_REASON, "human_review": True}
            )
        )


if __name__ == "__main__":
    unittest.main()
