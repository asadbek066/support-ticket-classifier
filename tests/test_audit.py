import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.audit import count_audit_logs, get_low_confidence_tickets, read_audit_logs


class AuditReadTests(unittest.TestCase):
    def test_limit_returns_newest_valid_entries_within_newest_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_file = root / "classifications-2026-09-09.jsonl"
            log_file.write_text(
                "".join(
                    json.dumps({"sequence": sequence}) + "\n"
                    for sequence in range(1, 5)
                ),
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

    def test_count_only_includes_valid_bounded_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_file = root / "classifications-2026-09-09.jsonl"
            log_file.write_text(
                json.dumps({"sequence": 1})
                + "\nnot-json\n"
                + json.dumps({"sequence": 2})
                + "\n"
                + json.dumps({"oversized": "x" * (128 * 1024)})
                + "\n",
                encoding="utf-8",
            )

            with patch("app.audit.AUDIT_LOG_DIR", root):
                count = count_audit_logs()

            self.assertEqual(count, 2)

    def test_manual_review_queue_includes_forced_review_and_ignores_malformed_classification(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_file = root / "classifications-2026-09-09.jsonl"
            log_file.write_text(
                "\n".join(
                    json.dumps(entry)
                    for entry in (
                        {
                            "ticket": {"subject": "security"},
                            "classification": {
                                "confidence": 0.99,
                                "human_review": True,
                            },
                        },
                        {
                            "ticket": {"subject": "billing"},
                            "classification": {
                                "confidence": 0.4,
                                "human_review": False,
                            },
                        },
                        {"classification": None},
                        {"classification": []},
                        {"classification": {"confidence": "not-a-number"}},
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("app.audit.AUDIT_LOG_DIR", root):
                entries = get_low_confidence_tickets(threshold=0.65, limit=10)

            self.assertEqual(len(entries), 2)
            self.assertEqual(
                [entry["ticket"]["subject"] for entry in entries],
                ["billing", "security"],
            )
            self.assertTrue(entries[1]["classification"]["human_review"])

    def test_manual_review_queue_finds_forced_review_beyond_recent_page(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_file = root / "classifications-2026-09-09.jsonl"
            entries = [
                {
                    "ticket": {"subject": "old-security"},
                    "classification": {"confidence": 0.99, "human_review": True},
                }
            ]
            entries.extend(
                {
                    "ticket": {"subject": f"routine-{index}"},
                    "classification": {"confidence": 0.99, "human_review": False},
                }
                for index in range(1000)
            )
            log_file.write_text(
                "\n".join(json.dumps(entry) for entry in entries) + "\n",
                encoding="utf-8",
            )

            with patch("app.audit.AUDIT_LOG_DIR", root):
                review_entries = get_low_confidence_tickets(threshold=0.65, limit=1)

            self.assertEqual(review_entries[0]["ticket"]["subject"], "old-security")


if __name__ == "__main__":
    unittest.main()
