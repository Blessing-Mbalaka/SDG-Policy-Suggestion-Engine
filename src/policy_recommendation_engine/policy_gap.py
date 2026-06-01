from __future__ import annotations

from policy_recommendation_engine.models import PolicyGap, Theme


class PolicyGapAnalyzer:
    def analyze(
        self,
        themes: tuple[Theme, ...],
        *,
        total_documents: int,
        policy_priorities: dict[str, float] | None = None,
    ) -> tuple[PolicyGap, ...]:
        priorities = {key.lower(): value for key, value in (policy_priorities or {}).items()}
        gaps: list[PolicyGap] = []

        for theme in themes:
            public_share = len(theme.document_indexes) / max(total_documents, 1)
            policy_share = self._matching_policy_share(theme, priorities)
            gap_score = max(public_share - policy_share, 0.0)
            gaps.append(
                PolicyGap(
                    theme=theme.name,
                    public_share=round(public_share, 4),
                    policy_share=round(policy_share, 4),
                    gap_score=round(gap_score, 4),
                    severity=self._severity(gap_score),
                )
            )

        return tuple(sorted(gaps, key=lambda gap: (-gap.gap_score, gap.theme)))

    def _matching_policy_share(self, theme: Theme, priorities: dict[str, float]) -> float:
        for keyword in theme.keywords:
            if keyword.lower() in priorities:
                return priorities[keyword.lower()]
        return 0.0

    def _severity(self, gap_score: float) -> str:
        if gap_score >= 0.3:
            return "high"
        if gap_score >= 0.15:
            return "medium"
        return "low"
