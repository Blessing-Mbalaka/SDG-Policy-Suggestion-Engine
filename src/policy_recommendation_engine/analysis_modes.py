from __future__ import annotations

from dataclasses import dataclass

from policy_recommendation_engine.embeddings import BertEmbeddingModel, HashEmbeddingModel
from policy_recommendation_engine.pipeline import PolicyIntelligencePipeline
from policy_recommendation_engine.preprocessing import SpacyTextPreprocessor, TextPreprocessor


@dataclass(frozen=True)
class AnalysisMode:
    key: str
    label: str
    description: str
    uses_spacy: bool = False
    uses_bert: bool = False


ANALYSIS_MODES = {
    "lightweight": AnalysisMode(
        key="lightweight",
        label="Fast local NLP",
        description="Regex tokenization, hashed vectors, and lexicon emotions.",
    ),
    "spacy": AnalysisMode(
        key="spacy",
        label="spaCy NLP",
        description="spaCy tokenization, lemmatization, sentence splitting, and named entities.",
        uses_spacy=True,
    ),
    "bert": AnalysisMode(
        key="bert",
        label="BERT semantic analysis",
        description="BERT-based SentenceTransformer embeddings for semantic theme grouping.",
        uses_bert=True,
    ),
    "spacy_bert": AnalysisMode(
        key="spacy_bert",
        label="spaCy + BERT",
        description="spaCy preprocessing plus transformer embeddings.",
        uses_spacy=True,
        uses_bert=True,
    ),
}


def build_pipeline(mode_key: str) -> PolicyIntelligencePipeline:
    mode = ANALYSIS_MODES.get(mode_key)
    if mode is None:
        raise ValueError(f"Unknown analysis mode: {mode_key}")

    preprocessor = SpacyTextPreprocessor() if mode.uses_spacy else TextPreprocessor()
    embedding_model = BertEmbeddingModel() if mode.uses_bert else HashEmbeddingModel()
    return PolicyIntelligencePipeline(preprocessor=preprocessor, embedding_model=embedding_model)
