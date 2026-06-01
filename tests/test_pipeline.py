from __future__ import annotations

import unittest
from datetime import datetime

from policy_recommendation_engine.ingestion import document_from_text
from policy_recommendation_engine.pipeline import PolicyIntelligencePipeline


class PipelineTests(unittest.TestCase):
    def test_pipeline_produces_themes_emotions_gaps_and_insights(self) -> None:
        documents = (
            document_from_text(
                "Water shortages are unbearable and the municipality is failing us.",
                source="forum",
                timestamp=datetime(2026, 6, 1),
            ),
            document_from_text(
                "No consistent access to water makes residents angry.",
                source="forum",
                timestamp=datetime(2026, 6, 2),
            ),
            document_from_text(
                "Healthcare delays make families afraid.",
                source="forum",
                timestamp=datetime(2026, 7, 1),
            ),
        )

        result = PolicyIntelligencePipeline().run(documents, policy_priorities={"water": 0.05})

        self.assertEqual(len(result.documents), 3)
        self.assertGreaterEqual(len(result.themes), 1)
        self.assertTrue(result.emotions_by_theme)
        self.assertTrue(result.policy_gaps)
        self.assertTrue(result.trends)
        self.assertTrue(result.insights)


if __name__ == "__main__":
    unittest.main()
