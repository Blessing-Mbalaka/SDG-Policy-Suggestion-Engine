from __future__ import annotations

import hashlib
import math

from policy_recommendation_engine.models import ProcessedDocument


Vector = tuple[float, ...]


class HashEmbeddingModel:
    """Deterministic local embedding stand-in for transformer embeddings."""

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions < 8:
            raise ValueError("Embedding dimensions must be at least 8.")
        self.dimensions = dimensions

    def embed(self, document: ProcessedDocument) -> Vector:
        vector = [0.0] * self.dimensions
        for token in document.tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return _normalize(vector)

    def embed_many(self, documents: tuple[ProcessedDocument, ...]) -> tuple[Vector, ...]:
        return tuple(self.embed(document) for document in documents)


class BertEmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/bert-base-nli-mean-tokens") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "SentenceTransformers is not installed. "
                "Run: .\\.venv\\Scripts\\python -m pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, document: ProcessedDocument) -> Vector:
        return self.embed_many((document,))[0]

    def embed_many(self, documents: tuple[ProcessedDocument, ...]) -> tuple[Vector, ...]:
        texts = [document.normalized_text for document in documents]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return tuple(tuple(float(value) for value in embedding) for embedding in embeddings)


def cosine_similarity(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _normalize(vector: list[float]) -> Vector:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return tuple(vector)
    return tuple(value / magnitude for value in vector)
