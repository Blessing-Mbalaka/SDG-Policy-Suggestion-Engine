from __future__ import annotations

import csv
from datetime import datetime
from io import BytesIO, StringIO

from policy_recommendation_engine.ingestion import document_from_text, documents_from_records
from policy_recommendation_engine.models import Document


def documents_from_upload(filename: str, content: bytes) -> tuple[Document, ...]:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"

    if suffix == "csv":
        text = decode_text_upload(content, suffix)
        return documents_from_records(csv.DictReader(StringIO(text)))
    if suffix in {"txt", "md"}:
        text = decode_text_upload(content, suffix)
        return (document_from_text(text, source=suffix, timestamp=datetime.now()),)
    if suffix == "pdf":
        text = extract_pdf_text(content)
        return (document_from_text(text, source="pdf", timestamp=datetime.now()),)

    raise ValueError(f"Unsupported upload format: .{suffix}")


def documents_from_pasted_text(text: str) -> tuple[Document, ...]:
    cleaned = text.strip()
    if not cleaned:
        return ()
    blocks = [block.strip() for block in cleaned.split("\n\n") if block.strip()]
    return tuple(document_from_text(block, source="manual", timestamp=datetime.now()) for block in blocks)


def decode_text_upload(content: bytes, suffix: str) -> str:
    try:
        text = content.decode("utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(f".{suffix} upload must be UTF-8 text. For PDFs, upload a .pdf file.") from exc
    if not text:
        raise ValueError(f".{suffix} upload is empty.")
    return text


def extract_pdf_text(content: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF upload requires pypdf. Run: .\\.venv\\Scripts\\python -m pip install pypdf") from exc

    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:  # noqa: BLE001 - pypdf raises several parser-specific errors.
        raise ValueError("Could not read the PDF file. It may be encrypted or damaged.") from exc

    page_text: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            page_text.append(text.strip())

    combined_text = "\n\n".join(page_text).strip()
    if not combined_text:
        raise ValueError("No readable text was found in the PDF. Scanned PDFs need OCR first.")
    return combined_text
