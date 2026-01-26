from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.ai_assistant.rag.embeddings import get_embeddings
from src.ai_assistant.core.config import config
from src.ai_assistant.core.logger import logger

client = QdrantClient(config.rag.db_url)

def get_vector_store() -> QdrantVectorStore:
    embeddings = get_embeddings()
    collection_name = "rag_store"

    collection_exists = client.collection_exists(collection_name)
    
    if not collection_exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=768,
                distance=models.Distance.COSINE
            ),
        )
        logger.info(f"Created new collection: {collection_name}")
    
    try:
        return QdrantVectorStore(
            collection_name=collection_name,
            embedding=embeddings,
            client=client,
        )
    except Exception as e:
        if "does not contain dense vector" in str(e):
            client.delete_collection(collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
            )
            return QdrantVectorStore(
                collection_name=collection_name,
                embedding=embeddings,
                client=client,
            )
        raise e