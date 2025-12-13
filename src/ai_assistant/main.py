from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.ai_assistant.api.v1 import router as api_v1_router
from src.ai_assistant.api.health import router as health_router
from src.ai_assistant.core.config import config
from src.ai_assistant.core.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.rag.create_docs_folder()
    logger.info("[LLM Service] Started")
    yield


def get_app():
    app = FastAPI(lifespan=lifespan, **config.app.model_dump())

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_v1_router)
    app.include_router(health_router)

    return app


app = get_app()
