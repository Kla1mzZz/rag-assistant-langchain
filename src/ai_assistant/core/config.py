from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings
from pydantic import BaseModel, field_validator


BASE_DIR = Path(__file__).resolve().parents[3]


class LLMConfig(BaseModel):
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.75
    top_k: int = 50
    top_p: float = 0.9

    api_key: str = ""

    prompts_dir: Path = BASE_DIR / "prompts"


class RAGConfig(BaseModel):
    embedding_model: str = "intfloat/multilingual-e5-base"
    persist_dir: Path = BASE_DIR / "chromadb"
    docs_folder: str = "docs"

    chunk_size: int = 800
    chunk_overlap: int = 150

    @field_validator("docs_folder", mode="before")
    @classmethod
    def validate_docs_folder(cls, v):
        path = Path(v) if not isinstance(v, Path) else v
        path.mkdir(parents=True, exist_ok=True)
        return path


class AppConfig(BaseModel):
    title: str = "LLM Service"
    version: str = "1.0.0"

    docs_url: str = "/docs"
    debug: bool = False

    host: str = "0.0.0.0"
    port: int = 8000

    cors_origins: List[str] = ["*"]
    cors_headers: List[str] = ["*"]
    cors_credentials: bool = True


class Config(BaseSettings):
    env: str = "dev"

    llm: LLMConfig = LLMConfig()
    rag: RAGConfig = RAGConfig()
    app: AppConfig = AppConfig()

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        extra = "ignore"


config = Config()
