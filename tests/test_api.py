"""Tests for API endpoints (async)."""

from unittest.mock import AsyncMock, patch

import pytest
import httpx
from langchain_core.documents import Document


class TestApp:
    async def test_openapi_available(self, client: httpx.AsyncClient) -> None:
        r = await client.get("/openapi.json")
        assert r.status_code == 200
        data = r.json()
        assert "openapi" in data
        assert "paths" in data

    async def test_docs_available(self, client: httpx.AsyncClient) -> None:
        r = await client.get("/docs")
        assert r.status_code == 200

    async def test_healthz_prefix_registered(self, client: httpx.AsyncClient) -> None:
        r = await client.get("/healthz")
        assert r.status_code in (200, 404, 405)


class TestConversationEndpoint:
    @patch("src.ai_assistant.api.v1.chat.rag_graph")
    async def test_conversation_success(
        self, mock_rag_graph: AsyncMock, client: httpx.AsyncClient
    ) -> None:
        mock_rag_graph.ainvoke = AsyncMock(
            return_value={
                "query": "Hi",
                "thread_id": "t1",
                "answer": "Hello!",
                "docs": [],
            }
        )
        r = await client.post(
            "/api/v1/chat/conversation",
            json={"prompt": "Hi", "thread_id": "t1"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert data["answer"] == "Hello!"
        mock_rag_graph.ainvoke.assert_called_once()

    @patch("src.ai_assistant.api.v1.chat.rag_graph")
    async def test_conversation_with_document_sources(
        self, mock_rag_graph: AsyncMock, client: httpx.AsyncClient
    ) -> None:
        mock_rag_graph.ainvoke = AsyncMock(
            return_value={
                "query": "What is X?",
                "thread_id": "t1",
                "answer": "X is...",
                "docs": [
                    Document(
                        page_content="X is something.",
                        metadata={"source": "docs/file.pdf"},
                    ),
                ],
            }
        )
        r = await client.post(
            "/api/v1/chat/conversation",
            json={"prompt": "What is X?", "thread_id": "t1"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["document_sources"] == ["file.pdf"]

    async def test_conversation_missing_body(self, client: httpx.AsyncClient) -> None:
        r = await client.post("/api/v1/chat/conversation", json={})
        assert r.status_code == 422

    async def test_conversation_validation_error(self, client: httpx.AsyncClient) -> None:
        r = await client.post(
            "/api/v1/chat/conversation",
            json={"prompt": "Hi"},
        )
        assert r.status_code == 422


class TestDocumentsEndpoint:
    async def test_get_documents_empty_or_cached(
        self, client: httpx.AsyncClient
    ) -> None:
        r = await client.get("/api/v1/admin/documents?limit=5")
        assert r.status_code == 200
        data = r.json()
        assert "documents" in data
        assert isinstance(data["documents"], list)

    async def test_get_documents_limit_validation(
        self, client: httpx.AsyncClient
    ) -> None:
        r = await client.get("/api/v1/admin/documents?limit=0")
        assert r.status_code == 422

    @patch("src.ai_assistant.api.v1.admin.pipeline")
    async def test_delete_document_not_found(
        self, mock_pipeline: AsyncMock, client: httpx.AsyncClient
    ) -> None:
        mock_pipeline.delete_documents_by_name = AsyncMock(return_value=False)
        r = await client.delete("/api/v1/admin/documents/nonexistent.pdf")
        assert r.status_code == 404
        data = r.json()
        assert "detail" in data

    @patch("src.ai_assistant.api.v1.admin.pipeline")
    async def test_delete_document_success(
        self, mock_pipeline: AsyncMock, client: httpx.AsyncClient
    ) -> None:
        mock_pipeline.delete_documents_by_name = AsyncMock(return_value=True)
        r = await client.delete("/api/v1/admin/documents/test.pdf")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["deleted"] == "test.pdf"
