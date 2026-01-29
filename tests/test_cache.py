"""Tests for cache module (key builders and async helpers with mocked Redis)."""

import hashlib

import pytest

from src.ai_assistant.core.cache import (
    CONVERSATION_KEY_PREFIX,
    RAG_RETRIEVE_KEY_PREFIX,
    OPTIMIZE_QUERY_KEY_PREFIX,
    GENERATE_KEY_PREFIX,
    DOCUMENTS_KEY_PREFIX,
    conversation_cache_key,
    rag_retrieve_cache_key,
    optimize_query_cache_key,
    generate_cache_key,
    get_json,
    set_json,
    delete_key,
)


class TestCacheKeyFunctions:
    def test_conversation_cache_key_prefix(self) -> None:
        key = conversation_cache_key("Hello")
        assert key.startswith(CONVERSATION_KEY_PREFIX)

    def test_conversation_cache_key_deterministic(self) -> None:
        k1 = conversation_cache_key("  Hello World  ")
        k2 = conversation_cache_key("hello world")
        assert k1 == k2

    def test_conversation_cache_key_normalized(self) -> None:
        key = conversation_cache_key("  Foo  ")
        normalized = "  foo  ".strip().lower()
        expected_hash = hashlib.sha256(normalized.encode()).hexdigest()[:32]
        assert key == f"{CONVERSATION_KEY_PREFIX}{expected_hash}"

    def test_rag_retrieve_cache_key_prefix(self) -> None:
        key = rag_retrieve_cache_key("query")
        assert key.startswith(RAG_RETRIEVE_KEY_PREFIX)

    def test_rag_retrieve_cache_key_deterministic(self) -> None:
        assert rag_retrieve_cache_key("q") == rag_retrieve_cache_key("  q  ")

    def test_optimize_query_cache_key_prefix(self) -> None:
        key = optimize_query_cache_key("query")
        assert key.startswith(OPTIMIZE_QUERY_KEY_PREFIX)

    def test_generate_cache_key_prefix(self) -> None:
        key = generate_cache_key("full prompt text")
        assert key.startswith(GENERATE_KEY_PREFIX)

    def test_generate_cache_key_no_normalization(self) -> None:
        # generate_cache_key uses prompt as-is (no strip/lower)
        k1 = generate_cache_key("A")
        k2 = generate_cache_key("a")
        assert k1 != k2


class TestCacheAsync:
    async def test_get_json_no_client_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.ai_assistant import core
        monkeypatch.setattr(core.config.config.cache, "enabled", False)
        # Reset client so _get_client returns None
        import src.ai_assistant.core.cache as cache_mod
        cache_mod._redis_client = None
        result = await get_json("any_key")
        assert result is None
        monkeypatch.setattr(core.config.config.cache, "enabled", True)

    async def test_set_json_no_client_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.ai_assistant import core
        monkeypatch.setattr(core.config.config.cache, "enabled", False)
        import src.ai_assistant.core.cache as cache_mod
        cache_mod._redis_client = None
        result = await set_json("key", {"a": 1}, 60)
        assert result is False
        monkeypatch.setattr(core.config.config.cache, "enabled", True)

    async def test_delete_key_no_client_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.ai_assistant import core
        monkeypatch.setattr(core.config.config.cache, "enabled", False)
        import src.ai_assistant.core.cache as cache_mod
        cache_mod._redis_client = None
        result = await delete_key("key")
        assert result is False
        monkeypatch.setattr(core.config.config.cache, "enabled", True)
