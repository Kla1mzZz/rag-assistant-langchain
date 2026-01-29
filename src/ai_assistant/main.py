import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.ai_assistant.api.v1 import router as api_v1_router
from src.ai_assistant.api.health import router as health_router
from src.ai_assistant.core.config import config
from src.ai_assistant.core.logger import logger
from src.ai_assistant.core.cache import close_redis


def _setup_langsmith() -> None:
    """Configure LangSmith tracing (env vars read by LangChain at runtime)."""
    os.environ["LANGCHAIN_TRACING_V2"] = str(config.langchain.tracing_v2).lower()
    if config.langchain.api_key and config.langchain.tracing_v2:
        os.environ["LANGCHAIN_API_KEY"] = config.langchain.api_key
        os.environ["LANGCHAIN_PROJECT"] = config.langchain.project
        logger.info(f"LangSmith tracing enabled (project: {config.langchain.project})")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.rag.create_docs_folder()
    _setup_langsmith()
    logger.info("[LLM Service] Started")
    yield
    await close_redis()


def get_app():
    app = FastAPI(lifespan=lifespan, **config.app.model_dump())

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.app.cors_origins,
        allow_headers=config.app.cors_headers,
        allow_methods=config.app.cors_methods,
    )

    app.include_router(api_v1_router)
    app.include_router(health_router)

    return app


app = get_app()
