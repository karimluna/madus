"""FastAPI application entrypoint.
Pre-warms PaddleOCR during startup so the first request doesn't
pay the 10+ second model loading penalty inside the request path.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from services.api.routes.analyze import router as analyze_router
from core.config import get_settings

logger = logging.getLogger(__name__)

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        from services.reasoning.graph.builder import build_graph
        _graph = build_graph()
    return _graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    logger.info("MADUS starting — LLM backend: %s", s.llm_backend)
    logger.info("Embedding backend: %s", s.embedding_backend)

    get_graph()

    # Pre-warm PaddleOCR in a thread so startup doesn't block
    import asyncio

    def _warm_ocr():
        try:
            from services.extraction.ocr import _get_ocr
            ocr = _get_ocr()
            logger.info("PaddleOCR warmed up successfully")
        except Exception as e:
            logger.warning("PaddleOCR warm-up failed (will retry on first request): %s", e)

    await asyncio.to_thread(_warm_ocr)

    yield


app = FastAPI(
    title="MADUS",
    description="Multi-Modal Agentic Document Understanding System",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(analyze_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}