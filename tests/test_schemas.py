"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from src.ai_assistant.schemas.chat import (
    ConversationRequest,
    ConversationResponse,
    ConversationResponseStream,
)
from src.ai_assistant.schemas.admin import (
    Document,
    DocumentGetResponse,
    DocumentUploadResponse,
    DocumentDeleteResponse,
)


class TestConversationRequest:
    def test_valid(self) -> None:
        r = ConversationRequest(prompt="Hello", thread_id="t1")
        assert r.prompt == "Hello"
        assert r.thread_id == "t1"

    def test_missing_prompt(self) -> None:
        with pytest.raises(ValidationError):
            ConversationRequest(thread_id="t1")

    def test_missing_thread_id(self) -> None:
        with pytest.raises(ValidationError):
            ConversationRequest(prompt="Hi")


class TestConversationResponse:
    def test_minimal(self) -> None:
        r = ConversationResponse(answer="Hi")
        assert r.answer == "Hi"
        assert r.document_sources is None

    def test_with_sources(self) -> None:
        r = ConversationResponse(
            answer="Based on doc1",
            document_sources=["doc1.pdf", "doc2.txt"],
        )
        assert r.document_sources == ["doc1.pdf", "doc2.txt"]


class TestConversationResponseStream:
    def test_defaults(self) -> None:
        r = ConversationResponseStream(type="chunk", content="hello")
        assert r.type == "chunk"
        assert r.content == "hello"
        assert r.tool is None


class TestDocument:
    def test_valid(self) -> None:
        d = Document(source="file.pdf", date="2025-01-01T00:00:00Z", size=1.5)
        assert d.source == "file.pdf"
        assert d.size == 1.5


class TestDocumentGetResponse:
    def test_empty_list(self) -> None:
        r = DocumentGetResponse(documents=[])
        assert r.documents == []

    def test_with_docs(self) -> None:
        r = DocumentGetResponse(
            documents=[
                Document(source="a.pdf", date="2025-01-01", size=0.1),
            ]
        )
        assert len(r.documents) == 1
        assert r.documents[0].source == "a.pdf"


class TestDocumentUploadResponse:
    def test_success(self) -> None:
        r = DocumentUploadResponse(success=True, filename="x.pdf")
        assert r.success is True
        assert r.filename == "x.pdf"


class TestDocumentDeleteResponse:
    def test_valid(self) -> None:
        r = DocumentDeleteResponse(success=True, deleted="x.pdf")
        assert r.success is True
        assert r.deleted == "x.pdf"
