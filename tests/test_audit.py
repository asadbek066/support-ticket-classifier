import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from app.audit import (
    count_audit_logs,
    count_low_confidence_tickets,
    get_low_confidence_tickets,
    log_classification,
    read_audit_logs,
)


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


class AuditWriteTests(unittest.TestCase):
    def test_log_classification_appends_to_the_utc_dated_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("app.audit.AUDIT_LOG_DIR", root):
                name = log_classification(
                    {"subject": "first"}, {"category": "Billing"}
                )
                log_classification({"subject": "second"}, {"category": "Billing"})

            self.assertRegex(name, r"^classifications-\d{4}-\d{2}-\d{2}\.jsonl$")
            lines = (root / name).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(
                json.loads(lines[0])["ticket"]["subject"], "first"
            )

    def test_log_classification_uses_one_clock_for_name_and_timestamp(self):
        first = datetime(2030, 1, 1, 23, 59, 59, tzinfo=UTC)
        second = datetime(2030, 1, 2, 0, 0, 0, tzinfo=UTC)

        class FakeDateTime:
            calls = 0

            @classmethod
            def now(cls, _tz=None):
                cls.calls += 1
                return first if cls.calls == 1 else second

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("app.audit.AUDIT_LOG_DIR", root),
                patch("app.audit.datetime", FakeDateTime),
            ):
                name = log_classification({"subject": "x"}, {"category": "c"})

            self.assertEqual(name, "classifications-2030-01-01.jsonl")
            entry = json.loads((root / name).read_text(encoding="utf-8"))
            self.assertTrue(entry["timestamp"].startswith("2030-01-01"))


class AuditCountTests(unittest.TestCase):
    def _entry(self, sequence, confidence=0.9, human_review=False):
        return {
            "sequence": sequence,
            "classification": {
                "confidence": confidence,
                "human_review": human_review,
            },
        }

    def test_counts_advance_with_appends(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_file = root / "classifications-2026-09-09.jsonl"
            log_file.write_text(
                json.dumps(self._entry(1)) + "\n", encoding="utf-8"
            )

            with patch("app.audit.AUDIT_LOG_DIR", root):
                self.assertEqual(count_audit_logs(), 1)
                self.assertEqual(count_low_confidence_tickets(0.65), 0)

                with log_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(self._entry(2, 0.2)) + "\n")
                    f.write("not-json\n")

                self.assertEqual(count_audit_logs(), 2)
                self.assertEqual(count_low_confidence_tickets(0.65), 1)

    def test_counts_are_summed_across_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "classifications-2026-09-08.jsonl").write_text(
                json.dumps(self._entry(1, 0.1)) + "\n", encoding="utf-8"
            )
            (root / "classifications-2026-09-09.jsonl").write_text(
                json.dumps(self._entry(2, human_review=True)) + "\n",
                encoding="utf-8",
            )

            with patch("app.audit.AUDIT_LOG_DIR", root):
                self.assertEqual(count_audit_logs(), 2)
                self.assertEqual(count_low_confidence_tickets(0.65), 2)

    def test_counts_reset_when_a_file_is_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_file = root / "classifications-2026-09-09.jsonl"
            log_file.write_text(
                "\n".join(json.dumps(self._entry(i)) for i in range(1, 4)) + "\n",
                encoding="utf-8",
            )

            with patch("app.audit.AUDIT_LOG_DIR", root):
                self.assertEqual(count_audit_logs(), 3)
                log_file.write_text(
                    json.dumps(self._entry(9)) + "\n", encoding="utf-8"
                )
                self.assertEqual(count_audit_logs(), 1)

    def test_counts_reset_after_copytruncate_regrowth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_file = root / "classifications-2026-09-09.jsonl"
            log_file.write_text(
                "\n".join(json.dumps(self._entry(i)) for i in range(1, 4)) + "\n",
                encoding="utf-8",
            )

            with patch("app.audit.AUDIT_LOG_DIR", root):
                self.assertEqual(count_audit_logs(), 3)
                # Prime the threshold cache too, so a stale offset would be
                # reused after the rewrite below.
                self.assertEqual(count_low_confidence_tickets(0.65), 0)

                # copytruncate rotation: same inode, truncated, then regrown
                # past the previously cached offsets with low-confidence
                # entries. A stale cache would count only the tail fragment.
                log_file.write_text(
                    "\n".join(
                        json.dumps(self._entry(i, 0.1))
                        for i in range(10, 20)
                    )
                    + "\n",
                    encoding="utf-8",
                )

                self.assertEqual(count_audit_logs(), 10)
                self.assertEqual(count_low_confidence_tickets(0.65), 10)

    def test_reads_span_files_newest_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "classifications-2026-09-08.jsonl").write_text(
                "\n".join(json.dumps(self._entry(i)) for i in (1, 2)) + "\n",
                encoding="utf-8",
            )
            (root / "classifications-2026-09-09.jsonl").write_text(
                "\n".join(json.dumps(self._entry(i)) for i in (3, 4)) + "\n",
                encoding="utf-8",
            )

            with patch("app.audit.AUDIT_LOG_DIR", root):
                self.assertEqual(
                    [entry["sequence"] for entry in read_audit_logs(limit=3)],
                    [4, 3, 2],
                )
                self.assertEqual(
                    [entry["sequence"] for entry in read_audit_logs(limit=10)],
                    [4, 3, 2, 1],
                )

    def test_reads_ignore_a_malformed_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_file = root / "classifications-2026-09-09.jsonl"
            log_file.write_text(
                json.dumps(self._entry(1)) + "\n" + '{"broken": ',
                encoding="utf-8",
            )

            with patch("app.audit.AUDIT_LOG_DIR", root):
                entries = read_audit_logs(limit=10)

            self.assertEqual([entry["sequence"] for entry in entries], [1])

    def test_unterminated_trailing_line_is_counted_once_terminated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_file = root / "classifications-2026-09-09.jsonl"
            log_file.write_text(json.dumps(self._entry(1)), encoding="utf-8")

            with patch("app.audit.AUDIT_LOG_DIR", root):
                self.assertEqual(count_audit_logs(), 0)
                with log_file.open("a", encoding="utf-8") as f:
                    f.write("\n")
                self.assertEqual(count_audit_logs(), 1)


if __name__ == "__main__":
    unittest.main()
