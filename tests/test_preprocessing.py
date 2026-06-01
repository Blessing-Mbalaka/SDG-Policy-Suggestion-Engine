from __future__ import annotations

import unittest

from policy_recommendation_engine.ingestion import document_from_text
from policy_recommendation_engine.preprocessing import TextPreprocessor


class PreprocessingTests(unittest.TestCase):
    def test_process_normalizes_segments_and_tokenizes(self) -> None:
        document = document_from_text("The WATER shortages are unbearable. We need action!")
        processed = TextPreprocessor().process(document)

        self.assertEqual(processed.normalized_text, "the water shortages are unbearable. we need action!")
        self.assertEqual(len(processed.sentences), 2)
        self.assertIn("water", processed.tokens)
        self.assertNotIn("the", processed.tokens)


if __name__ == "__main__":
    unittest.main()
