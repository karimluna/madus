"""SHA-256 content-adressed Redis cache. 
Two files producing the same digest would require a preimage 
attack which is computationally infeasible. So the hash is a 
content adress."""


import hashlib
import logging

import redis

from core.models import DocumentState
from core.config import get_settings

logger = logging.getLogger(__name__)


_ttl = 86400 # 24 hours


def _get_redis() -> redis.Redis:
    s = get_settings()
    return redis.Redis(host=s.redis_host, port=s.redis_port, decode_responses=True)


def _cache_key(pdf_path:str) -> str:
    with open(pdf_path, "rb") as f:
        return "madus:" + hashlib.sha256(f.read()).hexdigest() 


def get_cached(pdf_path: str) -> DocumentState | None:
    """Return cached state or `None`. Silently returns `None` if 
    Redis is unavailable."""
    try:
        r = _get_redis()
        raw = r.get(_cache_key(pdf_path))
        if raw:
            return DocumentState.model_validate_json(raw)
    except redis.ConnectionError:
        logger.warning("Redis unavailable, skipping cache lookup")
    return None


def set_cached(pdf_path: str, state: DocumentState) -> None:
    try: 
        r = _get_redis()
        r.setex(_cache_key(pdf_path), _ttl, state.model_dump_json())
    except redis.ConnectionError:
        logger.warning("Redis unavailable, skipping cache write")