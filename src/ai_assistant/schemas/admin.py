from pydantic import BaseModel


class Document(BaseModel):
    source: str
    date: str
    size: float


class DocumentGetResponse(BaseModel):
    documents: list[Document]


class DocumentUploadResponse(BaseModel):
    success: bool
    filename: str


class DocumentDeleteResponse(BaseModel):
    success: bool
    deleted: str
