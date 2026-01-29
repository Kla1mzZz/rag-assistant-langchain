"""Tests for application configuration."""

from pathlib import Path

import pytest
from pydantic_settings import BaseSettings

from src.ai_assistant.core.config import (
    BASE_DIR,
    Config,
    LLMConfig,
    RAGConfig,
    CacheConfig,
    AppConfig,
)


class TestLLMConfig:
    def test_defaults(self) -> None:
        c = LLMConfig()
        assert c.model_name == "gemini-2.5-flash"
        assert c.temperature == 0.75
        assert c.top_k == 50
        assert c.top_p == 0.9
        assert c.api_key == ""
        assert c.prompts_dir == BASE_DIR / "prompts"

    def test_prompts_dir_is_path(self) -> None:
        c = LLMConfig()
        assert isinstance(c.prompts_dir, Path)
        assert c.prompts_dir.name == "prompts"


class TestRAGConfig:
    def test_defaults(self) -> None:
        c = RAGConfig()
        assert c.db_url == "http://localhost:6333"
        assert c.embedding_model == "intfloat/multilingual-e5-base"
        assert c.docs_folder == "docs"
        assert c.chunk_size == 1500
        assert c.chunk_overlap == 150

    def test_create_docs_folder(self, tmp_path: Path) -> None:
        folder = tmp_path / "my_docs"
        c = RAGConfig(docs_folder=str(folder))
        assert not folder.exists()
        c.create_docs_folder()
        assert folder.exists() and folder.is_dir()
        c.create_docs_folder()  # idempotent
        assert folder.exists()


class TestCacheConfig:
    def test_defaults(self) -> None:
        c = CacheConfig()
        assert c.redis_url == "redis://localhost:6379/0"
        assert c.documents_ttl_seconds == 300
        assert c.conversation_ttl_seconds == 600
        assert c.rag_retrieve_ttl_seconds == 600
        assert c.optimize_query_ttl_seconds == 600
        assert c.generate_ttl_seconds == 600
        assert c.enabled is True


class TestAppConfig:
    def test_defaults(self) -> None:
        c = AppConfig()
        assert c.title == "LLM Service"
        assert c.version == "1.0.0"
        assert c.docs_url == "/docs"
        assert c.debug is False
        assert c.host == "0.0.0.0"
        assert c.port == 8000
        assert "*" in c.cors_origins
        assert c.cors_credentials is True


class TestConfig:
    def test_loads_nested_config(self) -> None:
        cfg = Config()
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.rag, RAGConfig)
        assert isinstance(cfg.cache, CacheConfig)
        assert isinstance(cfg.app, AppConfig)

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENV", "prod")
        monkeypatch.setenv("RAG__CHUNK_SIZE", "800")
        # Re-load settings (Config reads env at import; new instance may still use cached)
        cfg = Config()
        assert cfg.env == "prod"
        assert cfg.rag.chunk_size == 800
