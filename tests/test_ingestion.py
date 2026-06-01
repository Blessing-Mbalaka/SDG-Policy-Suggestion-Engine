from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from policy_recommendation_engine.ingestion import document_from_text, documents_from_records, load_documents


class IngestionTests(unittest.TestCase):
    def test_document_from_text_rejects_empty_text(self) -> None:
        with self.assertRaises(ValueError):
            document_from_text("   ")

    def test_documents_from_records_skips_empty_rows(self) -> None:
        documents = documents_from_records(
            [
                {"source": "youtube", "author": "user1", "text": "Water shortages are serious."},
                {"source": "reddit", "text": ""},
            ]
        )

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].source, "youtube")
        self.assertEqual(documents[0].author, "user1")

    def test_load_documents_reads_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.txt"
            path.write_text("Healthcare delays are frightening.", encoding="utf-8")

            documents = load_documents(path)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].source, "txt")


if __name__ == "__main__":
    unittest.main()
