from __future__ import annotations

from policy_recommendation_engine.models import EmotionSignal, PolicyGap, Theme


class InsightGenerator:
    def generate(
        self,
        themes: tuple[Theme, ...],
        emotions_by_theme: dict[str, EmotionSignal],
        policy_gaps: tuple[PolicyGap, ...],
    ) -> tuple[str, ...]:
        insights: list[str] = []
        for theme in themes:
            emotion = emotions_by_theme.get(theme.name)
            if emotion:
                insights.append(
                    f"{theme.name.title()} is linked to {emotion.dominant_emotion} sentiment "
                    f"with {emotion.intensity} intensity."
                )

        for gap in policy_gaps:
            if gap.gap_score > 0:
                insights.append(
                    f"{gap.theme.title()} shows a {gap.severity} policy gap "
                    f"({gap.public_share:.0%} public share vs {gap.policy_share:.0%} policy share)."
                )

        return tuple(insights)
