"""Extraction node: runs OCR, layout detection, and table parsing
sequentially. Results populate the text_chunks, images, and tables
fields in DocumentState.

Also indexes chunks in ChromaDB as a side effect so that semantic
retrieval is available when the text agent runs.
"""

import asyncio
import logging

from core.models import DocumentState
from core.embeddings import index_chunks
from services.extraction.ocr import extract_text_chunks
from services.extraction.vision import extract_images
from services.extraction.chunker import extract_tables

logger = logging.getLogger(__name__)


async def extraction_node(state: DocumentState) -> dict:
    """Extract all modalities from the PDF. Runs CPU-bound extraction
    in a thread pool to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()

    # Sequential extraction for reliability: parallel extraction
    # can cause OOM on large PDFs due to PaddleOCR + OpenCV memory usage
    text_chunks = await asyncio.to_thread(extract_text_chunks, state.pdf_path)
    # using asyncion.gather will be better if more RAM available...
    images = await asyncio.to_thread(extract_images, state.pdf_path)
    tables = await asyncio.to_thread(extract_tables, state.pdf_path)

    logger.info(
        "Extracted: %d text chunks, %d images, %d tables",
        len(text_chunks), len(images), len(tables),
    )

    # Index chunks for semantic retrieval, side effect but necessary
    # for the text agent's hybrid retrieval to work
    try:
        await asyncio.to_thread(index_chunks, state.doc_id, text_chunks)
    except Exception as e:
        logger.warning("ChromaDB indexing failed: %s", e)

    return {
        "text_chunks": text_chunks,
        "images": images,
        "tables": tables,
    }