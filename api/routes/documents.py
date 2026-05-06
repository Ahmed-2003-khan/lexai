from fastapi import APIRouter, Depends, Query, HTTPException
from api.services.document_service import DocumentService
from api.routes.auth import get_current_user
from api.config import get_settings

settings = get_settings()

from retriever.engine import DPRInferenceEngine
import os

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# Optional Embedder init if semantic search is hit
try:
    dpr_engine = DPRInferenceEngine(
        os.getenv("QUERY_ENCODER_PATH", "models/dpr/query_encoder.onnx"),
        os.getenv("PASSAGE_ENCODER_PATH", "models/dpr/passage_encoder.onnx"),
        os.getenv("TOKENIZER_PATH", "models/dpr/tokenizer")
    )
except:
    dpr_engine = None

doc_service = DocumentService(settings.DATABASE_URL, dpr_engine)

@router.get("/")
async def list_documents(page: int = 1, page_size: int = 20, jurisdiction: str = None, user: dict = Depends(get_current_user)):
    return await doc_service.get_all(page, page_size, jurisdiction)

@router.get("/search")
async def search_documents(q: str, jurisdiction: str = "PK", top_k: int = 10, user: dict = Depends(get_current_user)):
    return await doc_service.semantic_search(q, top_k, jurisdiction)

@router.get("/{doc_id}")
async def get_document(doc_id: str, user: dict = Depends(get_current_user)):
    doc = await doc_service.get_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@router.delete("/{doc_id}")
async def delete_document(doc_id: str, user: dict = Depends(get_current_user)):
    # Note: In real app, verify admin role here
    success = await doc_service.delete(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted"}