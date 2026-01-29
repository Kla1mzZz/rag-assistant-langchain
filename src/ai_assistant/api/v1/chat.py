from fastapi import APIRouter

from src.ai_assistant.graph.graph import rag_graph
from src.ai_assistant.graph.state import RAGState
from src.ai_assistant.schemas.chat import ConversationRequest, ConversationResponse
from src.ai_assistant.core.config import config
from src.ai_assistant.core.cache import (
    get_json,
    set_json,
    conversation_cache_key,
)


router = APIRouter()


@router.post("/conversation", response_model=ConversationResponse)
async def conversation(payload: ConversationRequest):
    cache_key = conversation_cache_key(payload.prompt)
    cached = await get_json(cache_key)
    if cached is not None:
        return ConversationResponse(**cached)

    state = RAGState(query=payload.prompt, thread_id=payload.thread_id)

    final_state = await rag_graph.ainvoke(state)

    documents_sources = []

    for doc in final_state["docs"]:
        if doc.metadata["source"].split("/")[-1] not in documents_sources:
            documents_sources.append(
                doc.metadata.get("source", "unknown").split("/")[-1]
            )

    response = ConversationResponse(
        answer=final_state["answer"], document_sources=documents_sources
    )
    await set_json(
        cache_key,
        response.model_dump(),
        config.cache.conversation_ttl_seconds,
    )
    return response


@router.post("/conversation/stream")
async def conversation_stream():
    pass
