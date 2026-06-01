from __future__ import annotations

from collections import Counter

from policy_recommendation_engine.embeddings import Vector, cosine_similarity
from policy_recommendation_engine.models import ProcessedDocument, Theme


class ThemeExtractor:
    def __init__(self, similarity_threshold: float = 0.18, max_keywords: int = 4) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_keywords = max_keywords

    def extract(self, documents: tuple[ProcessedDocument, ...], vectors: tuple[Vector, ...]) -> tuple[Theme, ...]:
        clusters: list[list[int]] = []
        centroids: list[Vector] = []

        for index, vector in enumerate(vectors):
            best_cluster = self._best_cluster(vector, centroids)
            if best_cluster is None:
                clusters.append([index])
                centroids.append(vector)
                continue
            clusters[best_cluster].append(index)
            centroids[best_cluster] = self._centroid(tuple(vectors[i] for i in clusters[best_cluster]))

        themes = [self._theme_from_cluster(cluster, documents) for cluster in clusters]
        return tuple(sorted(themes, key=lambda theme: (-len(theme.document_indexes), theme.name)))

    def _best_cluster(self, vector: Vector, centroids: list[Vector]) -> int | None:
        if not centroids:
            return None
        scored = [(index, cosine_similarity(vector, centroid)) for index, centroid in enumerate(centroids)]
        index, score = max(scored, key=lambda item: item[1])
        return index if score >= self.similarity_threshold else None

    def _centroid(self, vectors: tuple[Vector, ...]) -> Vector:
        dimensions = len(vectors[0])
        totals = [0.0] * dimensions
        for vector in vectors:
            for index, value in enumerate(vector):
                totals[index] += value
        return tuple(value / len(vectors) for value in totals)

    def _theme_from_cluster(self, cluster: list[int], documents: tuple[ProcessedDocument, ...]) -> Theme:
        counts: Counter[str] = Counter()
        for index in cluster:
            counts.update(documents[index].tokens)
        keywords = tuple(word for word, _ in counts.most_common(self.max_keywords))
        name = " / ".join(keywords[:2]) if keywords else "general concern"
        score = sum(counts.values()) / max(len(cluster), 1)
        return Theme(name=name, document_indexes=tuple(cluster), keywords=keywords, score=score)
