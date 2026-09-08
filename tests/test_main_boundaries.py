import unittest

from app.main import runtime_config, safe_api_url


class MainBoundaryTests(unittest.TestCase):
    def test_status_url_removes_credentials_and_query_data(self):
        self.assertEqual(
            safe_api_url("http://user:secret@example.test:11434/api/generate?token=private"),
            "http://example.test:11434/api/generate",
        )

    def test_runtime_config_rejects_non_string_provider_settings(self):
        with self.assertRaisesRegex(RuntimeError, "configuration unavailable"):
            runtime_config({"ollama": {"model": ["not-a-model"]}})


if __name__ == "__main__":
    unittest.main()
