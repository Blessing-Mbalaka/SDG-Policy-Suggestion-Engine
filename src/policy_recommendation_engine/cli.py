from __future__ import annotations

import argparse
import json
from datetime import datetime

from policy_recommendation_engine.ingestion import document_from_text, load_documents
from policy_recommendation_engine.pipeline import PolicyIntelligencePipeline


DEMO_TEXTS = (
    "The municipality has ignored water shortages for years and people are angry.",
    "No consistent access to water. The service failure is unbearable.",
    "Healthcare delays make families afraid during outbreaks.",
    "Public transport delays are frustrating and make workers late.",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the policy intelligence pipeline.")
    parser.add_argument("input", nargs="?", help="Path to a .txt, .md, or .csv input file.")
    parser.add_argument("--demo", action="store_true", help="Run with bundled demo documents.")
    args = parser.parse_args()

    if args.demo:
        documents = tuple(
            document_from_text(text, source="demo", timestamp=datetime(2026, 6, 1))
            for text in DEMO_TEXTS
        )
    elif args.input:
        documents = load_documents(args.input)
    else:
        parser.error("Provide an input path or use --demo.")

    result = PolicyIntelligencePipeline().run(
        documents,
        policy_priorities={"water": 0.05, "healthcare": 0.2, "transport": 0.1},
    )
    print(json.dumps(_result_to_dict(result), indent=2))


def _result_to_dict(result: object) -> dict[str, object]:
    return {
        "themes": [
            {"name": theme.name, "keywords": theme.keywords, "documents": theme.document_indexes}
            for theme in result.themes
        ],
        "emotions": {
            theme: {
                "dominant_emotion": signal.dominant_emotion,
                "intensity": signal.intensity,
                "scores": signal.scores,
            }
            for theme, signal in result.emotions_by_theme.items()
        },
        "policy_gaps": [
            {
                "theme": gap.theme,
                "public_share": gap.public_share,
                "policy_share": gap.policy_share,
                "gap_score": gap.gap_score,
                "severity": gap.severity,
            }
            for gap in result.policy_gaps
        ],
        "trends": result.trends,
        "insights": result.insights,
    }


if __name__ == "__main__":
    main()
