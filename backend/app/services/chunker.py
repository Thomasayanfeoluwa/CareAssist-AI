from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.pdf_loader import PdfPage

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

@dataclass(frozen=True)
class Chunk:
    "One retrievable unit. Carries everything needed for citation."
    chunk_id: str   # "{source}:{page}:{idx}" — deterministic, idempotent
    source: str
    title: str
    page: int
    text: str

def chunk_pages(
    pages: list[PdfPage],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap:  int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split each page into chunks, preserving page-level citation metadata.

    - Per-page: chunks never span page boundaries (citations stay unambiguous).
    - Empty pages are skipped (e.g. cover/title page).
    - Chunk IDs are deterministic so re-running this on the same PDF
      produces the same IDs and upsert is idempotent.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks: list[Chunk] = []
    skipped_empty_pages = 0

    for page in pages:
        if not page.text.strip():
            skipped_empty_pages += 1
            continue
            
        pieces = splitter.split_text(page.text)
        for idx, piece in enumerate(pieces):
            text = piece.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{page.source}:{page.page}:{idx}",
                    source=page.source,
                    title=page.title,
                    page=page.page,
                    text=text,
                )
            )

    logger.info(
        "chunked pages_in=%d chunks_out=%d skipped_empty_pages=%d",
        len(pages),
        len(chunks),
        skipped_empty_pages,
    )
    return chunks