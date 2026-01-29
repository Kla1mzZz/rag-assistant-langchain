"""Tests for RAG splitter."""

from langchain_core.documents import Document

from src.ai_assistant.rag.splitter import get_splitter


def test_get_splitter_returns_splitter() -> None:
    splitter = get_splitter()
    assert splitter is not None
    assert splitter._chunk_size > 0
    assert splitter._chunk_overlap >= 0


def test_split_documents_respects_chunk_size() -> None:
    splitter = get_splitter()
    # Create a document longer than chunk_size
    long_text = "a " * 800  # ~1600 chars
    docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]
    chunks = splitter.split_documents(docs)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk.page_content) <= splitter._chunk_size + splitter._chunk_overlap


def test_split_documents_preserves_metadata() -> None:
    splitter = get_splitter()
    docs = [
        Document(
            page_content="First paragraph.\n\nSecond paragraph.",
            metadata={"source": "x.pdf", "page": 1},
        )
    ]
    chunks = splitter.split_documents(docs)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.metadata.get("source") == "x.pdf"
        assert chunk.metadata.get("page") == 1


def test_split_short_document_single_chunk() -> None:
    splitter = get_splitter()
    short = Document(page_content="Short text.", metadata={})
    chunks = splitter.split_documents([short])
    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text."
