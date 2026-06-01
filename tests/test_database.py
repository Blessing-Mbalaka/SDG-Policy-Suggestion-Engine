from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from policy_recommendation_engine.database import (
    get_database_summary,
    initialize_database,
    list_analysis_runs,
    save_pipeline_result,
    seed_database,
)
from policy_recommendation_engine.ingestion import document_from_text
from policy_recommendation_engine.pipeline import PolicyIntelligencePipeline


class DatabaseTests(unittest.TestCase):
    def test_initialize_database_creates_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "test.sqlite"

            initialize_database(database_path)

            table_names = read_table_names(database_path)

        self.assertIn("analysis_runs", table_names)
        self.assertIn("documents", table_names)
        self.assertIn("themes", table_names)

    def test_save_pipeline_result_writes_analysis_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "test.sqlite"
            documents = (document_from_text("Water shortages are unbearable.", source="test"),)
            priorities = {"water": 0.05}
            result = PolicyIntelligencePipeline().run(documents, policy_priorities=priorities)

            run_id = save_pipeline_result(
                result,
                analysis_mode="lightweight",
                policy_priorities=priorities,
                database_path=database_path,
            )

            counts = read_counts(database_path)

        self.assertEqual(run_id, 1)
        self.assertEqual(counts["analysis_runs"], 1)
        self.assertEqual(counts["documents"], 1)
        self.assertGreaterEqual(counts["themes"], 1)

    def test_seed_database_creates_sample_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "seed.sqlite"

            run_id = seed_database(database_path)

            counts = read_counts(database_path)

        self.assertEqual(run_id, 1)
        self.assertEqual(counts["analysis_runs"], 1)
        self.assertEqual(counts["documents"], 3)

    def test_list_analysis_runs_returns_archive_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "archive.sqlite"
            seed_database(database_path)

            runs = list_analysis_runs(database_path)
            summary = get_database_summary(database_path)

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["id"], 1)
        self.assertEqual(runs[0]["document_count"], 3)
        self.assertGreaterEqual(len(runs[0]["themes"]), 1)
        self.assertEqual(summary["analysis_runs"], 1)


def read_table_names(database_path: Path) -> set[str]:
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    finally:
        connection.close()
    return {row[0] for row in rows}


def read_counts(database_path: Path) -> dict[str, int]:
    table_names = ("analysis_runs", "documents", "themes")
    counts: dict[str, int] = {}
    connection = sqlite3.connect(database_path)
    try:
        for table_name in table_names:
            row = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            counts[table_name] = int(row[0])
    finally:
        connection.close()
    return counts


if __name__ == "__main__":
    unittest.main()
