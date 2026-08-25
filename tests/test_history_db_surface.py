import os
import sqlite3
import tempfile
import unittest

from database import history_db


class HistoryDatabaseSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_path = history_db.DB_PATH
        history_db.DB_PATH = os.path.join(self.temp_dir.name, "history.db")

    def tearDown(self):
        history_db.DB_PATH = self.original_path
        self.temp_dir.cleanup()

    def test_existing_database_is_migrated_without_losing_rows(self):
        conn = sqlite3.connect(history_db.DB_PATH)
        conn.execute(
            """CREATE TABLE results (
                id INTEGER PRIMARY KEY AUTOINCREMENT, member TEXT NOT NULL,
                filename TEXT, fruit TEXT NOT NULL, label TEXT NOT NULL,
                confidence REAL NOT NULL, annotated_path TEXT, source TEXT,
                created_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            "INSERT INTO results (member, fruit, label, confidence, created_at) "
            "VALUES ('ensemble_ab', 'apple', 'ripe', 90.0, '2026-01-01')"
        )
        conn.commit()
        conn.close()

        history_db.init_db()
        row = history_db.get_by_id(1)
        self.assertEqual(row["label"], "ripe")
        self.assertIn("blemish_percentage", row)
        self.assertIsNone(row["blemish_percentage"])
        self.assertIn("marketability_status", row)
        self.assertIsNone(row["marketability_status"])

    def test_failed_analysis_persists_as_null_not_zero(self):
        history_db.init_db()
        history_db.log_result(
            member="ensemble_ab",
            fruit="apple",
            label="ripe",
            confidence=90.0,
            blemish_percentage=None,
            quality_grade=None,
        )
        row = history_db.get_by_id(1)
        self.assertIsNone(row["blemish_percentage"])
        self.assertIsNone(row["quality_grade"])

    def test_marketability_decision_is_persisted(self):
        history_db.init_db()
        history_db.log_result(
            member="ensemble_ab",
            fruit="apple",
            label="rotten",
            confidence=91.0,
            marketability_status="remove",
            dispatch_priority="remove",
            marketability_min_days=0,
            marketability_max_days=0,
            marketability_action="Do not market this fruit.",
            marketability_reliability="high",
            marketability_storage_assumption="whole fruit in cold storage",
        )
        row = history_db.get_by_id(1)
        self.assertEqual(row["marketability_status"], "remove")
        self.assertEqual(row["dispatch_priority"], "remove")
        self.assertEqual(row["marketability_min_days"], 0)
        self.assertEqual(row["marketability_max_days"], 0)

    def test_surface_analytics_ignore_failed_rows(self):
        history_db.init_db()
        history_db.log_result(
            member="ensemble_ab", fruit="apple", label="ripe", confidence=90.0,
            blemish_percentage=None, quality_grade=None,
        )
        history_db.log_result(
            member="ensemble_ab", fruit="apple", label="ripe", confidence=92.0,
            fruit_area_px=1000, blemish_area_px=50,
            blemish_percentage=5.0, quality_grade="Grade A",
        )
        stats = history_db.get_stats("ensemble_ab")
        self.assertEqual(stats["avg_blemish_percentage"], 5.0)
        self.assertEqual(stats["by_quality_grade"], {"Grade A": 1})


if __name__ == "__main__":
    unittest.main()
