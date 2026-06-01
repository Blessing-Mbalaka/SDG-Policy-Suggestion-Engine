from __future__ import annotations

import unittest

from policy_recommendation_engine.ingestion import document_from_text
from policy_recommendation_engine.network_graph import build_embedding_graph
from policy_recommendation_engine.pipeline import PolicyIntelligencePipeline


class NetworkGraphTests(unittest.TestCase):
    def test_build_embedding_graph_returns_nodes_edges_and_svg(self) -> None:
        documents = (
            document_from_text("Water shortages are unbearable."),
            document_from_text("No consistent access to water is frustrating."),
            document_from_text("Healthcare delays make families afraid."),
        )
        result = PolicyIntelligencePipeline().run(documents, policy_priorities={"water": 0.05})

        graph = build_embedding_graph(result)

        self.assertEqual(len(graph.nodes), 3)
        self.assertGreaterEqual(len(graph.edges), 1)
        self.assertIn("<svg", graph.svg)
        self.assertIn("Doc 1", graph.svg)


if __name__ == "__main__":
    unittest.main()
