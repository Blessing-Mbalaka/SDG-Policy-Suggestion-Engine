from __future__ import annotations

import csv
from datetime import datetime
from io import StringIO

from policy_recommendation_engine.ingestion import document_from_text, documents_from_records
from policy_recommendation_engine.models import Document


def documents_from_upload(filename: str, content: bytes) -> tuple[Document, ...]:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    text = content.decode("utf-8-sig").strip()
    if not text:
        return ()

    if suffix == "csv":
        return documents_from_records(csv.DictReader(StringIO(text)))
    if suffix in {"txt", "md"}:
        return (document_from_text(text, source=suffix, timestamp=datetime.now()),)

    raise ValueError(f"Unsupported upload format: .{suffix}")


def documents_from_pasted_text(text: str) -> tuple[Document, ...]:
    cleaned = text.strip()
    if not cleaned:
        return ()
    blocks = [block.strip() for block in cleaned.split("\n\n") if block.strip()]
    return tuple(document_from_text(block, source="manual", timestamp=datetime.now()) for block in blocks)
