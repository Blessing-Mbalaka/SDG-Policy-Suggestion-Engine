from __future__ import annotations

from policy_recommendation_engine.embeddings import HashEmbeddingModel
from policy_recommendation_engine.emotions import EmotionMapper
from policy_recommendation_engine.insights import InsightGenerator
from policy_recommendation_engine.models import Document, PipelineResult
from policy_recommendation_engine.policy_gap import PolicyGapAnalyzer
from policy_recommendation_engine.preprocessing import TextPreprocessor
from policy_recommendation_engine.themes import ThemeExtractor
from policy_recommendation_engine.trends import TrendAnalyzer


class PolicyIntelligencePipeline:
    def __init__(
        self,
        *,
        preprocessor: TextPreprocessor | None = None,
        embedding_model: HashEmbeddingModel | None = None,
        theme_extractor: ThemeExtractor | None = None,
        emotion_mapper: EmotionMapper | None = None,
        policy_gap_analyzer: PolicyGapAnalyzer | None = None,
        trend_analyzer: TrendAnalyzer | None = None,
        insight_generator: InsightGenerator | None = None,
    ) -> None:
        self.preprocessor = preprocessor or TextPreprocessor()
        self.embedding_model = embedding_model or HashEmbeddingModel()
        self.theme_extractor = theme_extractor or ThemeExtractor()
        self.emotion_mapper = emotion_mapper or EmotionMapper()
        self.policy_gap_analyzer = policy_gap_analyzer or PolicyGapAnalyzer()
        self.trend_analyzer = trend_analyzer or TrendAnalyzer()
        self.insight_generator = insight_generator or InsightGenerator()

    def run(
        self,
        documents: tuple[Document, ...],
        *,
        policy_priorities: dict[str, float] | None = None,
    ) -> PipelineResult:
        processed = tuple(self.preprocessor.process(document) for document in documents)
        vectors = self.embedding_model.embed_many(processed)
        themes = self.theme_extractor.extract(processed, vectors)
        emotions = self.emotion_mapper.map_themes(themes, processed)
        gaps = self.policy_gap_analyzer.analyze(
            themes,
            total_documents=len(processed),
            policy_priorities=policy_priorities,
        )
        trends = self.trend_analyzer.summarize(themes, processed)
        insights = self.insight_generator.generate(themes, emotions, gaps)

        return PipelineResult(
            documents=processed,
            themes=themes,
            emotions_by_theme=emotions,
            policy_gaps=gaps,
            trends=trends,
            insights=insights,
        )
