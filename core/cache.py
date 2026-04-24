"""SHA-256 content-adressed Redis cache.
Two files producing the same digest would require a preimage
attack which is computationally infeasible. So the hash is a
content adress."""

import hashlib
import logging
import asyncio

import redis

from core.models import DocumentState
from core.config import get_settings

logger = logging.getLogger(__name__)


_ttl = 86400  # 24 hours


def _get_redis() -> redis.Redis:
    s = get_settings()
    return redis.Redis(host=s.redis_host, port=s.redis_port, decode_responses=True)


def _cache_key(pdf_path: str, question: str = "") -> str:
    """Derive a cache key from PDF content hash + question hash.
    keying on content (not path) means the same file uploaded under
    different names still hits cache."""
    with open(pdf_path, "rb") as f:
        pdf_hash = hashlib.sha256(f.read()).hexdigest()
    q_hash = hashlib.sha256(question.encode()).hexdigest()[:16]
    return f"madus:{pdf_hash}:{q_hash}"


def _get_cached_sync(pdf_path: str, question: str) -> DocumentState | None:
    try:
        r = _get_redis()
        raw = r.get(_cache_key(pdf_path, question))
        if raw:
            return DocumentState.model_validate_json(raw)
    except redis.ConnectionError:
        logger.warning("Redis unavailable, skipping cache lookup")
    return None


def _set_cached_sync(pdf_path: str, question: str, state: DocumentState) -> None:
    try:
        r = _get_redis()
        r.setex(_cache_key(pdf_path, question), _ttl, state.model_dump_json())
    except redis.ConnectionError:
        logger.warning("Redis unavailable, skipping cache write")


async def get_cached(pdf_path: str, question: str = "") -> DocumentState | None:
    return await asyncio.to_thread(_get_cached_sync, pdf_path, question)


async def set_cached(pdf_path: str, question: str, state: DocumentState) -> None:
    await asyncio.to_thread(_set_cached_sync, pdf_path, question, state)
