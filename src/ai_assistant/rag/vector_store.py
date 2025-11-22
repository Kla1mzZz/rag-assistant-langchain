from langchain_chroma import Chroma
from src.ai_assistant.rag.embeddings import get_embeddings
from src.ai_assistant.core.config import config


def get_vector_store() -> Chroma:
    embeddings = get_embeddings()

    return Chroma(
        collection_name="rag_store",
        embedding_function=embeddings,
        persist_directory=config.rag.persist_dir,
    )
