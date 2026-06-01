from __future__ import annotations

import unittest

from policy_recommendation_engine.upload import documents_from_pasted_text, documents_from_upload
from policy_recommendation_engine.web import parse_analysis_mode, parse_policy_priorities


class UploadTests(unittest.TestCase):
    def test_pasted_text_splits_blank_line_documents(self) -> None:
        documents = documents_from_pasted_text("Water shortages are bad.\n\nHealthcare delays are scary.")

        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].source, "manual")

    def test_csv_upload_reads_text_column(self) -> None:
        content = b"source,author,text\nforum,user1,Transport delays are frustrating.\n"

        documents = documents_from_upload("comments.csv", content)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].source, "forum")

    def test_pdf_like_bytes_are_not_decoded_as_text(self) -> None:
        content = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\nnot a full pdf"

        with self.assertRaises(ValueError) as context:
            documents_from_upload("broken.pdf", content)

        self.assertIn("Could not read the PDF", str(context.exception))

    def test_binary_text_upload_gets_clear_error(self) -> None:
        content = b"\xe2 bad utf8"

        with self.assertRaises(ValueError) as context:
            documents_from_upload("comments.txt", content)

        self.assertIn("must be UTF-8 text", str(context.exception))

    def test_policy_priorities_default_when_empty(self) -> None:
        priorities = parse_policy_priorities("")

        self.assertIn("water", priorities)

    def test_analysis_mode_defaults_to_lightweight(self) -> None:
        self.assertEqual(parse_analysis_mode(""), "lightweight")

    def test_analysis_mode_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            parse_analysis_mode("magic")


if __name__ == "__main__":
    unittest.main()
