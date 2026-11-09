"""One-time index builder: PDF -> chunks -> vectors -> Pinecone.

Usage (from backend/):
    python -m app.scripts.build_index

Idempotent: re-running on the same PDF upserts the same chunk_ids, so
Pinecone ends up with the same vectors (upsert overwrites by id).
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from app.config import get_settings
from app.logging_setup import configure_logging
from app.services.chunker import chunk_pages
from app.services.embedder import Embedder
from app.services.indexer import PineconeIndexer
from app.services.pdf_loader import load_pdf

logger = logging.getLogger("build_index")

PDF_PATH = Path(__file__).resolve().parents[3] / "Data" / "Medical_book.pdf"


def main() -> int:
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)

    if not PDF_PATH.exists():
        logger.error("pdf_missing path=%s", PDF_PATH)
        return 1

    logger.info("build_start pdf=%s index=%s", PDF_PATH, settings.PINECONE_INDEX)

    t0 = time.perf_counter()

    logger.info("phase=load")
    pages = load_pdf(PDF_PATH)
    logger.info("loaded pages=%d elapsed=%.1fs", len(pages), time.perf_counter() - t0)

    logger.info("phase=chunk")
    t1 = time.perf_counter()
    chunks = chunk_pages(pages)
    logger.info(
        "chunked chunks=%d elapsed=%.1fs",
        len(chunks),
        time.perf_counter() - t1,
    )

    logger.info("phase=embed")
    t2 = time.perf_counter()
    embedder = Embedder()
    vectors = embedder.embed([c.text for c in chunks])
    logger.info(
        "embedded count=%d dim=%d elapsed=%.1fs",
        len(vectors),
        len(vectors[0]) if vectors else 0,
        time.perf_counter() - t2,
    )

    logger.info("phase=upsert")
    t3 = time.perf_counter()
    result = PineconeIndexer(settings).build(chunks, vectors)
    logger.info(
        "upserted count=%d index_created=%s elapsed=%.1fs",
        result.upserted,
        result.index_created,
        time.perf_counter() - t3,
    )

    logger.info("build_done total_elapsed=%.1fs", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())