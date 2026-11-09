from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfPage:
    """One page of a PDF with citation metadata.

    page is 1-indexed (user-facing). title is derived from the filename so
    citations read 'PDF: Medical book, page 42' rather than a path.
    """
    source: str          # original filename, "Medical_book.pdf"
    title: str           # human-readable, "Medical book"
    page: int            # 1-indexed page number
    text: str            # extracted text (may be "" if the page is scanned or empty)


def load_pdf(path: str | Path) -> list[PdfPage]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")
    if p.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF: {p}")

    reader = PdfReader(str(p))
    title = p.stem.replace("_", " ").strip() or p.stem
    pages: list[PdfPage] = []

    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 — per-page failure should not abort the whole load
            logger.warning(
                "pdf_page_extract_failed source=%s page=%d error_type=%s",
                p.name,
                idx,
                type(exc).__name__,
            )
            text = ""
        pages.append(PdfPage(source=p.name, title=title, page=idx, text=text))

    logger.info("pdf_loaded source=%s pages=%d", p.name, len(pages))
    return pages


def load_pdfs(directory: str | Path) -> list[PdfPage]:
    """Load every *.pdf in a directory, sorted by name for deterministic order."""
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {d}")

    pages: list[PdfPage] = []
    for pdf_path in sorted(d.glob("*.pdf")):
        pages.extend(load_pdf(pdf_path))
    return pages