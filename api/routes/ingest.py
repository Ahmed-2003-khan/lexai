import os
import tempfile
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from api.middleware.auth import get_current_user
from api.config import get_settings
from ingestion.pipeline import IngestionPipeline
from ingestion.loader import DocumentLoader

router = APIRouter(tags=["ingestion"])
settings = get_settings()

pipeline = IngestionPipeline(db_url=settings.DATABASE_URL)

class UrlIngestRequest(BaseModel):
    url: str
    title: str
    source: str
    jurisdiction: str = "PK"
    doc_type: str = "article"

@router.post("/ingest/file")
async def api_ingest_file(
    file: UploadFile = File(...),
    title: str = Form(...),
    source: str = Form(...),
    jurisdiction: str = Form("PK"),
    doc_type: str = Form(...),
    user: dict = Depends(get_current_user)
):
    """Uploads and ingests a file into the vector database."""
    valid_exts = [".pdf", ".txt", ".docx"]
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in valid_exts:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        stats = await pipeline.ingest_file(temp_path, title, source, jurisdiction, doc_type)
        return stats
    finally:
        os.unlink(temp_path)

@router.post("/ingest/url")
async def api_ingest_url(request: UrlIngestRequest, user: dict = Depends(get_current_user)):
    """Fetches text from a URL and ingests it."""
    content = DocumentLoader.load_from_url(request.url)
    if not content:
        raise HTTPException(status_code=400, detail="Could not extract content from URL")
        
    stats = await pipeline.ingest_text(
        content, request.title, request.source, request.jurisdiction, request.doc_type
    )
    return stats

@router.get("/ingest/stats")
async def api_ingest_stats():
    """Returns database statistics. (No auth required)."""
    return await pipeline.get_stats()