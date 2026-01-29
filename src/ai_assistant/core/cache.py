import hashlib
import json
from typing import Any

from src.ai_assistant.core.config import config
from src.ai_assistant.core.logger import logger

_redis_client: Any = None

DOCUMENTS_KEY_PREFIX = "documents:"
CONVERSATION_KEY_PREFIX = "conversation:"
RAG_RETRIEVE_KEY_PREFIX = "rag_retrieve:"
OPTIMIZE_QUERY_KEY_PREFIX = "optimize_query:"
GENERATE_KEY_PREFIX = "generate:"


def _get_client():
    global _redis_client
    if not config.cache.enabled:
        return None
    if _redis_client is None:
        try:
            import redis.asyncio as redis

            _redis_client = redis.from_url(
                config.cache.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        except Exception as e:
            logger.warning(f"Redis cache disabled: {e}")
            return None
    return _redis_client


async def get_json(key: str) -> dict | list | None:
    """Get JSON value from cache. Returns None on miss or error."""
    client = _get_client()
    if not client:
        return None
    try:
        raw = await client.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.debug(f"Cache get error for {key}: {e}")
        return None


async def set_json(key: str, value: dict | list, ttl_seconds: int) -> bool:
    """Set JSON value with TTL. Returns True on success."""
    client = _get_client()
    if not client:
        return False
    try:
        await client.set(key, json.dumps(value), ex=ttl_seconds)
        return True
    except Exception as e:
        logger.debug(f"Cache set error for {key}: {e}")
        return False


async def delete_key(key: str) -> bool:
    """Delete single key. Returns True on success."""
    client = _get_client()
    if not client:
        return False
    try:
        await client.delete(key)
        return True
    except Exception as e:
        logger.debug(f"Cache delete error for {key}: {e}")
        return False


async def delete_documents_cache() -> None:
    """Invalidate documents list and RAG retrieve cache (after add/delete document)."""
    client = _get_client()
    if not client:
        return
    try:
        keys = []
        async for key in client.scan_iter(match=f"{DOCUMENTS_KEY_PREFIX}*"):
            keys.append(key)
        async for key in client.scan_iter(match=f"{RAG_RETRIEVE_KEY_PREFIX}*"):
            keys.append(key)
        if keys:
            await client.delete(*keys)
            logger.debug(f"Invalidated documents and RAG cache: {len(keys)} keys")
    except Exception as e:
        logger.debug(f"Cache invalidation error: {e}")


def conversation_cache_key(prompt: str) -> str:
    """Stable cache key for conversation by prompt (normalized)."""
    normalized = prompt.strip().lower()
    h = hashlib.sha256(normalized.encode()).hexdigest()[:32]
    return f"{CONVERSATION_KEY_PREFIX}{h}"


def rag_retrieve_cache_key(query: str) -> str:
    """Stable cache key for RAG retrieve by query."""
    normalized = query.strip().lower()
    h = hashlib.sha256(normalized.encode()).hexdigest()[:32]
    return f"{RAG_RETRIEVE_KEY_PREFIX}{h}"


def optimize_query_cache_key(query: str) -> str:
    """Stable cache key for optimized query by original query."""
    normalized = query.strip().lower()
    h = hashlib.sha256(normalized.encode()).hexdigest()[:32]
    return f"{OPTIMIZE_QUERY_KEY_PREFIX}{h}"


def generate_cache_key(prompt: str) -> str:
    """Stable cache key for LLM generation by full prompt (context + query)."""
    h = hashlib.sha256(prompt.encode()).hexdigest()[:32]
    return f"{GENERATE_KEY_PREFIX}{h}"


async def delete_rag_retrieve_cache() -> None:
    """Invalidate all RAG retrieve cache (after add/delete document)."""
    client = _get_client()
    if not client:
        return
    try:
        keys = []
        async for key in client.scan_iter(match=f"{RAG_RETRIEVE_KEY_PREFIX}*"):
            keys.append(key)
        if keys:
            await client.delete(*keys)
            logger.debug(f"Invalidated RAG retrieve cache: {len(keys)} keys")
    except Exception as e:
        logger.debug(f"RAG cache invalidation error: {e}")


async def close_redis() -> None:
    """Close Redis connection (e.g. on app shutdown)."""
    global _redis_client
    if _redis_client:
        try:
            await _redis_client.aclose()
        except Exception as e:
            logger.debug(f"Redis close: {e}")
        _redis_client = None
