from fastapi import APIRouter

from src.ai_assistant.graph.graph import rag_graph
from src.ai_assistant.graph.state import RAGState
from src.ai_assistant.schemas.chat import ConversationRequest, ConversationResponse


router = APIRouter()


@router.post("/conversation", response_model=ConversationResponse)
async def conversation(payload: ConversationRequest):
    state = RAGState(query=payload.prompt, thread_id=payload.thread_id)

    final_state = await rag_graph.ainvoke(state)

    documents_sources = []

    for doc in final_state["docs"]:
        if doc.metadata["source"].split("/")[-1] not in documents_sources:
            documents_sources.append(
                doc.metadata.get("source", "unknown").split("/")[-1]
            )

    return ConversationResponse(
        answer=final_state["answer"], document_sources=documents_sources
    )


@router.post("/conversation/stream")
async def conversation_stream():
    pass
