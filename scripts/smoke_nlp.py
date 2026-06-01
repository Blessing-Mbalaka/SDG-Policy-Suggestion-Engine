from __future__ import annotations

from policy_recommendation_engine.analysis_modes import build_pipeline
from policy_recommendation_engine.ingestion import document_from_text


def main() -> None:
    documents = (
        document_from_text("Water shortages in Johannesburg are unbearable and residents are angry."),
        document_from_text("Healthcare delays in Cape Town make families afraid."),
    )

    for mode in ("spacy", "bert", "spacy_bert"):
        result = build_pipeline(mode).run(documents, policy_priorities={"water": 0.05, "healthcare": 0.2})
        print(f"{mode}: {len(result.themes)} themes, {len(result.insights)} insights")
        print(f"  first theme: {result.themes[0].name}")
        print(f"  entities: {result.documents[0].named_entities}")


if __name__ == "__main__":
    main()
