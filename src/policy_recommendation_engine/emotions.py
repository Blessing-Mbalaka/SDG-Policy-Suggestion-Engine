from __future__ import annotations

from collections import Counter

from policy_recommendation_engine.models import EmotionSignal, ProcessedDocument, Theme


EMOTION_LEXICON = {
    "anger": {"angry", "ignored", "corrupt", "failure", "failing", "unbearable", "outrage", "broken"},
    "fear": {"afraid", "fear", "unsafe", "risk", "danger", "outbreak", "threat"},
    "frustration": {"delay", "delays", "waiting", "shortage", "shortages", "unstable", "problem", "complaint"},
    "hopelessness": {"hopeless", "nothing", "never", "years", "unemployment", "poverty"},
    "distrust": {"distrust", "lied", "promise", "promises", "transparent", "accountability"},
    "optimism": {"improve", "better", "progress", "hope", "working", "satisfied"},
}


class EmotionMapper:
    def map_theme(self, theme: Theme, documents: tuple[ProcessedDocument, ...]) -> EmotionSignal:
        scores: Counter[str] = Counter()
        for index in theme.document_indexes:
            token_set = set(documents[index].tokens)
            for emotion, words in EMOTION_LEXICON.items():
                scores[emotion] += len(token_set & words)

        if not scores or max(scores.values(), default=0) == 0:
            return EmotionSignal(dominant_emotion="neutral", intensity="low", scores={})

        emotion, score = scores.most_common(1)[0]
        return EmotionSignal(
            dominant_emotion=emotion,
            intensity=self._intensity(score, len(theme.document_indexes)),
            scores=dict(scores),
        )

    def map_themes(
        self,
        themes: tuple[Theme, ...],
        documents: tuple[ProcessedDocument, ...],
    ) -> dict[str, EmotionSignal]:
        return {theme.name: self.map_theme(theme, documents) for theme in themes}

    def _intensity(self, score: int, document_count: int) -> str:
        density = score / max(document_count, 1)
        if density >= 2:
            return "high"
        if density >= 1:
            return "medium"
        return "low"
