import os
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_community.document_compressors import FlashrankRerank
from qdrant_client.http import models

from src.ai_assistant.rag.splitter import get_splitter
from src.ai_assistant.rag.vector_store import get_vector_store
from src.ai_assistant.core.logger import logger
from src.ai_assistant.core.config import config


class RAGPipeline:
    def __init__(self, store=None):
        self.splitter = get_splitter()
        self.vector_store = store or get_vector_store()
        self.re_ranker = FlashrankRerank()

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

    async def get_documents(self, limit: int = 10) -> list[dict[str, Any]]:
        try:
            records, _ = self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                limit=limit * 5,
                with_payload=True,
                with_vectors=False,
            )

            seen_sources = set()
            unique_documents = []

            for record in records:
                payload = record.payload or {}
                metadata = payload.get("metadata", {})

                source = metadata.get("source")

                if source and source not in seen_sources:
                    seen_sources.add(source)
                    unique_documents.append(
                        {
                            "id": record.id,
                            "metadata": metadata,
                        }
                    )

                if len(unique_documents) >= limit:
                    break

            return unique_documents

        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}")
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
                logger.warning(
                    f"File {file_path} not found on disk, but cleanup continued."
                )

            return True

        except Exception as e:
            logger.error(f"Error during deletion process for {name}: {str(e)}")
            return False

    async def index_documents(self, docs: list[Document], batch_size: int = 30) -> bool:
        try:
            texts = self.splitter.split_documents(docs)
            current_time = datetime.now(timezone.utc).isoformat()

            final_docs = []
            ids = []

            for chunk in texts:
                content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()
                source_path = chunk.metadata.get("source", "unknown")
                chunk_id = f"{source_path}_{content_hash}"
                deterministic_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

                if source_path and os.path.exists(source_path):
                    file_size_bytes = os.path.getsize(source_path)
                    file_size_kb = round(file_size_bytes / 1024, 2)
                    chunk.metadata["file_size"] = file_size_kb / 1000
                else:
                    chunk.metadata["file_size"] = 0

                chunk.metadata["chunk_id"] = chunk_id
                chunk.metadata["created_at"] = current_time

                final_docs.append(chunk)
                ids.append(deterministic_uuid)

            for i in range(0, len(final_docs), batch_size):
                batch_docs = final_docs[i : i + batch_size]
                batch_ids = ids[i : i + batch_size]

                await self.vector_store.aadd_documents(
                    documents=batch_docs, ids=batch_ids
                )
                logger.info(f"Indexed batch {i // batch_size + 1}")

            return True

        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            return False

    async def retrieve(self, query: str, k: int = 3) -> list[Document]:
        base_retriever = self.vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 20, "fetch_k": 50}
        )

        self.re_ranker.top_n = k
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.re_ranker, base_retriever=base_retriever
        )

        # documents = await self.vector_store.amax_marginal_relevance_search(
        #     query, k=k, fetch_k=k * 5
        # )

        return await compression_retriever.ainvoke(query)
