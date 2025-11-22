from fastapi import APIRouter

from src.ai_assistant.api.v1.admin import router as admin_router
from src.ai_assistant.api.v1.chat import router as chat_router

router = APIRouter(prefix="/api/v1", tags=["v1"])

router.include_router(chat_router, prefix="/chat")
router.include_router(admin_router, prefix="/admin")
