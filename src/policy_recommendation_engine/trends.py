from __future__ import annotations

from collections import defaultdict

from policy_recommendation_engine.models import ProcessedDocument, Theme


class TrendAnalyzer:
    def summarize(
        self,
        themes: tuple[Theme, ...],
        documents: tuple[ProcessedDocument, ...],
    ) -> dict[str, dict[str, int]]:
        trends: dict[str, dict[str, int]] = {}
        for theme in themes:
            buckets: defaultdict[str, int] = defaultdict(int)
            for index in theme.document_indexes:
                timestamp = documents[index].document.timestamp
                bucket = timestamp.strftime("%Y-%m") if timestamp else "undated"
                buckets[bucket] += 1
            trends[theme.name] = dict(sorted(buckets.items()))
        return trends
