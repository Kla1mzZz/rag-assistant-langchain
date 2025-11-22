from langchain_huggingface import HuggingFaceEmbeddings
from src.ai_assistant.core.config import config


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=config.rag.embedding_model,
    )
