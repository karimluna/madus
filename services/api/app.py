"""FastAPI application entrypoint."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from services.api.routes.analyze import router as analyze_router
from services.reasoning.graph.builder import build_graph
from core.config import get_settings

logger = logging.getLogger(__name__)

# Compiled graph — built once, reused across requests
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: verify connections. Shutdown: clean up."""
    s = get_settings()
    logger.info("MADUS starting — LLM backend: %s", s.llm_backend)
    logger.info("Embedding backend: %s", s.embedding_backend)
    # Pre-build graph on startup to catch config errors early
    get_graph()
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