import hashlib
import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from app import audit
from app.audit import (
    AuditSettings,
    count_audit_logs,
    count_low_confidence_tickets,
    get_low_confidence_tickets,
    log_classification,
    prune_audit_files,
    read_audit_logs,
    settings_from_config,
)

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
TODAY = "classifications-2026-10-08.jsonl"


class AuditBoundsBase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        patches = [
            patch("app.audit.AUDIT_LOG_DIR", self.root),
            patch.object(audit, "_SETTINGS", AuditSettings()),
            patch.object(audit, "_DROPPED_BY_DAY", {}),
            patch.object(audit, "_DROPPED_TOTAL", 0),
            patch.object(audit, "_WARNED_DAYS", set()),
            patch.object(audit, "_LAST_PRUNE", None),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(self.temp.cleanup)

    def make(self, name, size=10):
        path = self.root / name
        path.write_bytes(b"x" * size)
        return path


class TruncationTests(AuditBoundsBase):
    def test_text_is_truncated_with_hash_and_length(self):
        text = "a" * 5000
        with patch("app.audit.datetime") as clock:
            clock.now.return_value = NOW
            log_classification(
                {"subject": "Hi", "description": text}, {"category": "c"}
            )
        record = json.loads((self.root / TODAY).read_text())
        self.assertEqual(len(record["ticket"]["description"]), 300)
        self.assertEqual(record["ticket"]["subject"], "Hi")
        digest = record["ticket_digest"]["description"]
        self.assertEqual(digest["chars"], 5000)
        self.assertEqual(digest["sha256"], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(record["truncated_fields"], ["description"])

    def test_full_text_switch_keeps_everything(self):
        audit.configure(AuditSettings(store_full_text=True))
        text = "b" * 5000
        log_classification({"description": text}, {"category": "c"})
        record = json.loads(next(self.root.glob("*.jsonl")).read_text())
        self.assertEqual(record["ticket"]["description"], text)
        self.assertEqual(record["truncated_fields"], [])
        self.assertEqual(record["ticket_digest"]["description"]["chars"], 5000)


class PruneTests(AuditBoundsBase):
    def test_old_files_are_deleted_but_recent_and_current_kept(self):
        self.make("classifications-2026-08-01.jsonl")
        self.make("classifications-2026-09-20.jsonl")
        self.make(TODAY)
        removed = prune_audit_files(AuditSettings(retention_days=30), now=NOW)
        self.assertEqual(removed, ["classifications-2026-08-01.jsonl"])
        self.assertTrue((self.root / TODAY).exists())
        self.assertTrue((self.root / "classifications-2026-09-20.jsonl").exists())

    def test_current_file_survives_even_when_over_every_bound(self):
        self.make(TODAY, 5000)
        removed = prune_audit_files(
            AuditSettings(retention_days=1, max_total_bytes=1), now=NOW
        )
        self.assertEqual(removed, [])
        self.assertTrue((self.root / TODAY).exists())

    def test_current_file_is_kept_even_when_expired_by_the_clock(self):
        # "now" is far in the future, so today's file is older than retention.
        future = datetime(2027, 1, 1, tzinfo=UTC)
        current = "classifications-2026-10-08.jsonl"
        self.make(current)
        self.make("classifications-2026-10-07.jsonl")
        removed = prune_audit_files(
            AuditSettings(retention_days=1), now=future, current_name=current
        )
        self.assertEqual(removed, ["classifications-2026-10-07.jsonl"])
        self.assertTrue((self.root / current).exists())

    def test_size_cap_removes_oldest_first(self):
        for day in ("05", "06", "07"):
            self.make(f"classifications-2026-10-{day}.jsonl", 400)
        self.make(TODAY, 400)
        removed = prune_audit_files(AuditSettings(max_total_bytes=900), now=NOW)
        self.assertEqual(
            removed,
            ["classifications-2026-10-05.jsonl", "classifications-2026-10-06.jsonl"],
        )

    def test_only_pattern_matching_regular_files_are_touched(self):
        outside = self.root.parent / f"{self.root.name}-outside.jsonl"
        outside.write_text("keep")
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        keep = [
            self.make("notes.txt", 5000),
            self.make("classifications-2020-01-01.jsonl.bak", 5000),
            self.make("classifications-old.jsonl", 5000),
            self.make("other-2020-01-01.jsonl", 5000),
        ]
        (self.root / "classifications-2020-01-01").mkdir()
        os.symlink(outside, self.root / "classifications-2020-01-02.jsonl")
        prune_audit_files(
            AuditSettings(retention_days=1, max_total_bytes=1), now=NOW
        )
        for path in keep:
            self.assertTrue(path.exists(), path.name)
        self.assertTrue((self.root / "classifications-2020-01-01").is_dir())
        self.assertTrue((self.root / "classifications-2020-01-02.jsonl").is_symlink())
        self.assertEqual(outside.read_text(), "keep")

    def test_prune_runs_once_per_interval_during_writes(self):
        audit.configure(AuditSettings(prune_interval_seconds=3600))
        with patch("app.audit.prune_audit_files") as prune:
            log_classification({"subject": "a"}, {"category": "c"})
            log_classification({"subject": "b"}, {"category": "c"})
            self.assertEqual(prune.call_count, 1)
            with patch("app.audit.monotonic", return_value=audit._LAST_PRUNE + 3601):
                log_classification({"subject": "c"}, {"category": "c"})
            self.assertEqual(prune.call_count, 2)

    def test_write_prunes_expired_files_without_touching_current(self):
        self.make("classifications-2001-01-01.jsonl")
        log_classification({"subject": "a"}, {"category": "c"})
        self.assertFalse((self.root / "classifications-2001-01-01.jsonl").exists())
        self.assertEqual(len(list(self.root.glob("*.jsonl"))), 1)


class DailyCapTests(AuditBoundsBase):
    def test_records_beyond_cap_are_dropped_counted_and_warned_once(self):
        audit.configure(AuditSettings(max_daily_bytes=600))
        with self.assertLogs("app.audit", level="WARNING") as logs:
            for index in range(10):
                log_classification({"subject": f"t{index}"}, {"category": "c"})
        written = (self.root / next(self.root.glob("*.jsonl")).name).read_text()
        kept = len(written.splitlines())
        self.assertGreaterEqual(kept, 1)
        self.assertLess(kept, 10)
        self.assertEqual(audit.drop_stats()["dropped_today"], 10 - kept)
        warnings = [r for r in logs.records if "daily_cap" in r.getMessage()]
        self.assertEqual(len(warnings), 1)
        self.assertEqual(
            oct(os.stat(self.root / next(self.root.glob("*.jsonl")).name).st_mode & 0o777),
            oct(0o600),
        )


class ReaderCompatibilityTests(AuditBoundsBase):
    def test_readers_handle_old_full_text_and_new_records(self):
        old = {
            "timestamp": "2026-10-07T00:00:00+00:00",
            "ticket": {"subject": "old", "description": "z" * 5000},
            "classification": {"category": "c", "confidence": 0.1, "human_review": True},
        }
        (self.root / "classifications-2026-10-07.jsonl").write_text(
            json.dumps(old) + "\n"
        )
        log_classification(
            {"subject": "new", "description": "y" * 5000},
            {"category": "c", "confidence": 0.2, "human_review": True},
        )
        subjects = [e["ticket"]["subject"] for e in read_audit_logs()]
        self.assertEqual(subjects, ["new", "old"])
        self.assertEqual(count_audit_logs(), 2)
        self.assertEqual(count_low_confidence_tickets(0.65), 2)
        self.assertEqual(len(get_low_confidence_tickets(0.65, 10)), 2)


class SettingsTests(unittest.TestCase):
    def test_defaults_when_section_absent(self):
        self.assertEqual(settings_from_config({}), AuditSettings())

    def test_values_are_read_and_validated(self):
        settings = settings_from_config(
            {"audit": {"max_text_chars": 50, "retention_days": 7, "max_total_mb": 1}}
        )
        self.assertEqual(settings.max_text_chars, 50)
        self.assertEqual(settings.retention_days, 7)
        self.assertEqual(settings.max_total_bytes, 1024 * 1024)
        for bad in (
            {"max_text_chars": -1},
            {"retention_days": 0},
            {"store_full_text": "yes"},
            {"max_daily_mb": True},
            {"mystery": 1},
        ):
            with self.assertRaises(ValueError, msg=str(bad)):
                settings_from_config({"audit": bad})


if __name__ == "__main__":
    unittest.main()
