from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    success: bool
    filename: str


class DocumentGetResponse(BaseModel):
    documents: list[str]


class DocumentDeleteResponse(BaseModel):
    success: bool
    deleted: str
