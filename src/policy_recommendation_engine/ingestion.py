from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable

from policy_recommendation_engine.models import Document


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md"}


def document_from_text(
    text: str,
    *,
    source: str = "manual",
    author: str | None = None,
    timestamp: datetime | None = None,
) -> Document:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Document text cannot be empty.")
    return Document(source=source, author=author, timestamp=timestamp, text=cleaned)


def documents_from_records(records: Iterable[dict[str, object]]) -> tuple[Document, ...]:
    documents: list[Document] = []
    for record in records:
        text = str(record.get("text", "")).strip()
        if not text:
            continue
        timestamp = _parse_timestamp(record.get("timestamp"))
        documents.append(
            Document(
                source=str(record.get("source", "record")),
                author=_optional_str(record.get("author")),
                timestamp=timestamp,
                text=text,
                metadata={k: v for k, v in record.items() if k not in {"source", "author", "timestamp", "text"}},
            )
        )
    return tuple(documents)


def load_documents(path: str | Path) -> tuple[Document, ...]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    suffix = file_path.suffix.lower()
    if suffix in SUPPORTED_TEXT_EXTENSIONS:
        return (document_from_text(file_path.read_text(encoding="utf-8"), source=suffix.lstrip(".")),)
    if suffix == ".csv":
        with file_path.open(newline="", encoding="utf-8") as csv_file:
            return documents_from_records(csv.DictReader(csv_file))

    raise ValueError(f"Unsupported input format: {suffix or '<none>'}")


def _parse_timestamp(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _optional_str(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
