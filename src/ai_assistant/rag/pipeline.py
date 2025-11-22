import os
from typing import Any
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document

from src.ai_assistant.rag.splitter import get_splitter
from src.ai_assistant.rag.vector_store import get_vector_store
from src.ai_assistant.core.logger import logger
from src.ai_assistant.core.config import config


class RAGPipeline:
    def __init__(self, store=None):
        self.splitter = get_splitter()
        self.vector_store = store or get_vector_store()

    async def extract_document(self, filename: str, file_path: str) -> list[Document] | None:
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif filename.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                loader = None
            return await loader.aload()
        except Exception as e:
            logger.error(f"Error extracting document {filename}: {str(e)}")
            return None

    async def get_documents(self, limit: int = 5) -> Any:
        documents = self.vector_store._collection.get(
            limit=limit-1,
            include=["documents", "metadatas"],
        )

        return documents

    async def delete_documents_by_name(self, name: str) -> bool:
        documents = self.vector_store._collection.get(
            where={"source": f"{config.rag.docs_folder}/{name}"},
            include=["documents", "metadatas"],
        )
        if not documents["ids"]:
            return False

        try:
            await self.vector_store.adelete(documents["ids"])
            os.remove(f"{config.rag.docs_folder}/{name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {name}: {str(e)}")
            return False

    async def index_documents(self, docs) -> bool:
        texts = self.splitter.split_documents(docs)
        try:
            await self.vector_store.aadd_documents(texts)
            return True
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            return False

    async def retrieve(self, query: str, k: int = 4) -> list[Document]:
        documents = await self.vector_store.amax_marginal_relevance_search(
            query, k=k, fetch_k=k * 5
        )

        return documents
