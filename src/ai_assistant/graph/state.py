from pydantic import BaseModel, Field
from langchain_core.documents import Document


class RAGState(BaseModel):
    query: str | None = None
    thread_id: str | None = None
    query_optimized: str | None = None
    use_rag: bool = True
    docs: list[Document] = Field(default_factory=list)
    prompt: str | None = None
    answer: str = None
