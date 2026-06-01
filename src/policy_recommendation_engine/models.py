from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Document:
    source: str
    text: str
    author: str | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessedDocument:
    document: Document
    normalized_text: str
    sentences: tuple[str, ...]
    tokens: tuple[str, ...]
    named_entities: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Theme:
    name: str
    document_indexes: tuple[int, ...]
    keywords: tuple[str, ...]
    score: float


@dataclass(frozen=True)
class EmotionSignal:
    dominant_emotion: str
    intensity: str
    scores: dict[str, int]


@dataclass(frozen=True)
class PolicyGap:
    theme: str
    public_share: float
    policy_share: float
    gap_score: float
    severity: str


@dataclass(frozen=True)
class PipelineResult:
    documents: tuple[ProcessedDocument, ...]
    embedding_vectors: tuple[tuple[float, ...], ...]
    themes: tuple[Theme, ...]
    emotions_by_theme: dict[str, EmotionSignal]
    policy_gaps: tuple[PolicyGap, ...]
    trends: dict[str, dict[str, int]]
    insights: tuple[str, ...]
