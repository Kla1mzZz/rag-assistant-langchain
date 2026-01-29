from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parents[3]


class LLMConfig(BaseModel):
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.75
    top_k: int = 50
    top_p: float = 0.9

    api_key: str = ""

    prompts_dir: Path = BASE_DIR / "prompts"


class RAGConfig(BaseModel):
    db_url: str = "http://localhost:6333"
    embedding_model: str = "intfloat/multilingual-e5-base"
    docs_folder: str = "docs"

    chunk_size: int = 1500
    chunk_overlap: int = 150

    def create_docs_folder(self):
        Path(self.docs_folder).mkdir(parents=True, exist_ok=True)


class CacheConfig(BaseModel):
    redis_url: str = "redis://localhost:6379/0"
    documents_ttl_seconds: int = 300  # 5 min
    conversation_ttl_seconds: int = 600  # 10 min
    rag_retrieve_ttl_seconds: int = 600  # 10 min
    optimize_query_ttl_seconds: int = 600  # 10 min
    generate_ttl_seconds: int = 600  # 10 min
    enabled: bool = True


class AppConfig(BaseModel):
    title: str = "LLM Service"
    version: str = "1.0.0"

    docs_url: str = "/docs"
    debug: bool = False

    host: str = "0.0.0.0"
    port: int = 8000

    cors_origins: List[str] = ["*"]
    cors_headers: List[str] = ["*"]
    cors_methods: List[str] = ["*"]
    cors_credentials: bool = True


class Config(BaseSettings):
    env: str = "dev"

    llm: LLMConfig = LLMConfig()
    rag: RAGConfig = RAGConfig()
    cache: CacheConfig = CacheConfig()
    app: AppConfig = AppConfig()

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        extra = "ignore"


config = Config()
