from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ai_assistant.core.config import config


def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=config.rag.chunk_size,
        chunk_overlap=config.rag.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
