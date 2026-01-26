from fastapi import APIRouter, UploadFile, File, HTTPException, status, Query

from src.ai_assistant.rag.pipeline import RAGPipeline
from src.ai_assistant.core.logger import logger
from src.ai_assistant.core.config import config
from src.ai_assistant.schemas.admin import (
    DocumentUploadResponse,
    DocumentDeleteResponse,
    DocumentGetResponse,
    Document
)


router = APIRouter()
pipeline = RAGPipeline()


@router.post("/documents", response_model=DocumentUploadResponse)
async def add_documents(file: UploadFile = File(...)):
    file_path = f"{config.rag.docs_folder}/{file.filename}"

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save file",
        )

    document = await pipeline.extract_document(file.filename, file_path)

    if not document:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Failed to process document",
        )

    index_document = await pipeline.index_documents(document)

    if not index_document:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to index document",
        )

    return DocumentUploadResponse(success=True, filename=file.filename)


@router.get("/documents", response_model=DocumentGetResponse)
async def get_documents(limit: int = Query(5, ge=1)):
    documents_data = await pipeline.get_documents(limit=limit)
    documents = []

    for doc in documents_data:
        documents.append(Document(source=doc['metadata'].get("source", "unknown").split("/")[-1], 
                                  date=doc['metadata'].get("created_at", "none"),
                                  size=doc['metadata'].get("file_size", 0.0)))

    return DocumentGetResponse(documents=documents)


@router.delete("/documents/{doc_name}", response_model=DocumentDeleteResponse)
async def delete_document(doc_name: str):
    is_deleted = await pipeline.delete_documents_by_name(doc_name)

    if not is_deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_name}' not found",
        )

    return DocumentDeleteResponse(success=True, deleted=doc_name)
