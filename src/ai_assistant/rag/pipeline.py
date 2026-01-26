import os
from datetime import datetime, timezone
from typing import Any
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from qdrant_client.http import models

from src.ai_assistant.rag.splitter import get_splitter
from src.ai_assistant.rag.vector_store import get_vector_store
from src.ai_assistant.core.logger import logger
from src.ai_assistant.core.config import config


class RAGPipeline:
    def __init__(self, store=None):
        self.splitter = get_splitter()
        self.vector_store = store or get_vector_store()

    async def extract_document(
        self, filename: str, file_path: str
    ) -> list[Document] | None:
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

    async def get_documents(self, limit: int = 5) -> list[dict[str, Any]]:
        try:
            records, _ = self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            documents = []
            for record in records:
                payload = record.payload or {}
                documents.append({
                    "id": record.id,
                    "metadata": payload.get("metadata", {}),
                })

            return documents

        except Exception as e:
            logger.error(f"Ошибка при получении документов: {str(e)}")
            return []

    async def delete_documents_by_name(self, name: str) -> bool:
        file_path = f"{config.rag.docs_folder}/{name}"
        try:
            delete_result = self.vector_store.client.delete(
                collection_name=self.vector_store.collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=file_path),
                        )
                    ]
                ),
            )

            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"File {file_path} deleted from file system.")
            else:
                logger.warning(f"File {file_path} not found on disk, but cleanup continued.")

            return True

        except Exception as e:
            logger.error(f"Error during deletion process for {name}: {str(e)}")
            return False

    async def index_documents(self, docs) -> bool:
        try:
            texts = self.splitter.split_documents(docs)
            current_time = datetime.now(timezone.utc).isoformat()
            
            for chunk in texts:
                chunk.metadata["created_at"] = current_time
                
                source_path = chunk.metadata.get("source")
                if source_path and os.path.exists(source_path):
                    file_size_bytes = os.path.getsize(source_path)
                    file_size_kb = round(file_size_bytes / 1024, 2)
                    chunk.metadata["file_size"] = file_size_kb
                else:
                    chunk.metadata["file_size"] = 0

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
