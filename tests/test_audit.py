import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.audit import read_audit_logs


class AuditReadTests(unittest.TestCase):
    def test_limit_returns_newest_valid_entries_within_newest_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_file = root / "classifications-2026-09-09.jsonl"
            log_file.write_text(
                "".join(json.dumps({"sequence": sequence}) + "\n" for sequence in range(1, 5)),
                encoding="utf-8",
            )

            with patch("app.audit.AUDIT_LOG_DIR", root):
                entries = read_audit_logs(limit=2)

            self.assertEqual([entry["sequence"] for entry in entries], [4, 3])

    def test_oversized_line_is_skipped_without_poisoning_later_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_file = root / "classifications-2026-09-09.jsonl"
            log_file.write_text(
                json.dumps({"oversized": "x" * (128 * 1024)})
                + "\n"
                + json.dumps({"sequence": 2})
                + "\n",
                encoding="utf-8",
            )

            with patch("app.audit.AUDIT_LOG_DIR", root):
                entries = read_audit_logs(limit=10)

            self.assertEqual(entries, [{"sequence": 2}])


if __name__ == "__main__":
    unittest.main()
