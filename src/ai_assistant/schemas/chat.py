from pydantic import BaseModel


class ConversationRequest(BaseModel):
    prompt: str


class ConversationResponseStream(BaseModel):
    type: str
    content: str | None = None
    tool: str | None = None


class ConversationResponse(BaseModel):
    answer: str
    document_sources: list[str] | None = None
