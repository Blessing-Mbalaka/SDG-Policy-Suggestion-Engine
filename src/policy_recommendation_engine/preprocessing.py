from __future__ import annotations

import re

from policy_recommendation_engine.models import Document, ProcessedDocument


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "this",
    "to",
    "we",
    "with",
}

TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z'-]+")
WHITESPACE_PATTERN = re.compile(r"\s+")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


class TextPreprocessor:
    def process(self, document: Document) -> ProcessedDocument:
        normalized = self.normalize(document.text)
        sentences = tuple(sentence.strip() for sentence in SENTENCE_PATTERN.split(normalized) if sentence.strip())
        tokens = tuple(token for token in self.tokenize(normalized) if token not in STOP_WORDS)
        return ProcessedDocument(
            document=document,
            normalized_text=normalized,
            sentences=sentences or (normalized,),
            tokens=tokens,
        )

    def normalize(self, text: str) -> str:
        text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
        text = WHITESPACE_PATTERN.sub(" ", text)
        return text.strip().lower()

    def tokenize(self, text: str) -> tuple[str, ...]:
        return tuple(match.group(0).lower() for match in TOKEN_PATTERN.finditer(text))


class SpacyTextPreprocessor(TextPreprocessor):
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError("spaCy is not installed. Run: .\\.venv\\Scripts\\python -m pip install spacy") from exc

        try:
            self.nlp = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed. "
                f"Run: .\\.venv\\Scripts\\python -m spacy download {model_name}"
            ) from exc

    def process(self, document: Document) -> ProcessedDocument:
        normalized = self.normalize(document.text)
        doc = self.nlp(normalized)
        tokens = tuple(
            (token.lemma_ or token.text).lower()
            for token in doc
            if not token.is_space and not token.is_punct and not token.is_stop and token.is_alpha
        )
        sentences = tuple(sentence.text.strip() for sentence in doc.sents if sentence.text.strip())
        named_entities = tuple((entity.text, entity.label_) for entity in doc.ents)
        return ProcessedDocument(
            document=document,
            normalized_text=normalized,
            sentences=sentences or (normalized,),
            tokens=tokens,
            named_entities=named_entities,
        )
