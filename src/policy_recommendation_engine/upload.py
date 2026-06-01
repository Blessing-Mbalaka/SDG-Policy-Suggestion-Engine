from __future__ import annotations

import csv
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from uuid import uuid4

from policy_recommendation_engine.ingestion import documents_from_records
from policy_recommendation_engine.models import Document


def default_media_uploads_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "media" / "uploads"


def save_uploaded_file(filename: str, content: bytes, media_dir: str | Path | None = None) -> Path:
    upload_dir = Path(media_dir) if media_dir is not None else default_media_uploads_path()
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = make_safe_filename(filename)
    saved_path = upload_dir / f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid4().hex}_{safe_name}"
    saved_path.write_bytes(content)
    return saved_path


def make_safe_filename(filename: str) -> str:
    name = Path(filename).name.strip() or "upload"
    safe_characters: list[str] = []
    for character in name:
        if character.isalnum() or character in {".", "-", "_"}:
            safe_characters.append(character)
        else:
            safe_characters.append("_")
    return "".join(safe_characters)


def documents_from_upload(
    filename: str,
    content: bytes,
    saved_path: str | Path | None = None,
) -> tuple[Document, ...]:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    metadata = upload_metadata(filename, saved_path)

    if suffix == "csv":
        text = decode_text_upload(content, suffix)
        records = []
        for row_number, row in enumerate(csv.DictReader(StringIO(text)), start=1):
            row["source"] = row.get("source") or "csv"
            row["upload_filename"] = filename
            row["upload_path"] = str(saved_path) if saved_path else ""
            row["upload_row"] = row_number
            records.append(row)
        return documents_from_records(records)
    if suffix in {"txt", "md"}:
        text = decode_text_upload(content, suffix)
        return documents_from_text_blocks(text, source=suffix, base_metadata=metadata)
    if suffix == "pdf":
        return documents_from_pdf_pages(content, filename=filename, saved_path=saved_path)

    raise ValueError(f"Unsupported upload format: .{suffix}")


def documents_from_pasted_text(text: str) -> tuple[Document, ...]:
    cleaned = text.strip()
    if not cleaned:
        return ()
    return documents_from_text_blocks(cleaned, source="manual")


def documents_from_text_blocks(
    text: str,
    *,
    source: str,
    base_metadata: dict[str, object] | None = None,
) -> tuple[Document, ...]:
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    documents: list[Document] = []
    for index, block in enumerate(blocks, start=1):
        metadata = dict(base_metadata or {})
        metadata["block_number"] = index
        documents.append(
            Document(
                source=source,
                text=block,
                timestamp=datetime.now(),
                metadata=metadata,
            )
        )
    return tuple(documents)


def upload_metadata(filename: str, saved_path: str | Path | None) -> dict[str, object]:
    metadata: dict[str, object] = {"upload_filename": filename}
    if saved_path is not None:
        metadata["upload_path"] = str(saved_path)
    return metadata


def decode_text_upload(content: bytes, suffix: str) -> str:
    try:
        text = content.decode("utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(f".{suffix} upload must be UTF-8 text. For PDFs, upload a .pdf file.") from exc
    if not text:
        raise ValueError(f".{suffix} upload is empty.")
    return text


def documents_from_pdf_pages(
    content: bytes,
    *,
    filename: str,
    saved_path: str | Path | None = None,
) -> tuple[Document, ...]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF upload requires pypdf. Run: .\\.venv\\Scripts\\python -m pip install pypdf") from exc

    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:  # noqa: BLE001 - pypdf raises several parser-specific errors.
        raise ValueError("Could not read the PDF file. It may be encrypted or damaged.") from exc

    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            metadata = upload_metadata(filename, saved_path)
            metadata["page_number"] = page_number
            documents.append(
                Document(
                    source="pdf",
                    text=text.strip(),
                    timestamp=datetime.now(),
                    metadata=metadata,
                )
            )

    if not documents:
        raise ValueError("No readable text was found in the PDF. Scanned PDFs need OCR first.")
    return tuple(documents)
