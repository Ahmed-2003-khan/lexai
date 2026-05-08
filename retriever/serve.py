import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

from retriever.engine import DPRInferenceEngine

app = FastAPI(title="LexAI Retriever Microservice")

# Global engine instance
engine = None
db_engine = None


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: List[float]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    jurisdiction: Optional[str] = None
    doc_type: Optional[str] = None


class SearchResult(BaseModel):
    rank: int
    score: float
    title: str
    source: str
    jurisdiction: str
    doc_type: str
    chunk_index: int
    content_preview: str


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResult]


@app.on_event("startup")
async def startup_event():
    global engine, db_engine
    try:
        query_model = os.getenv("QUERY_ENCODER_PATH", "models/dpr/query_encoder.onnx")
        passage_model = os.getenv("PASSAGE_ENCODER_PATH", "models/dpr/passage_encoder.onnx")
        tokenizer_path = os.getenv("TOKENIZER_PATH", "models/dpr/tokenizer")

        if os.path.exists(query_model) and os.path.exists(tokenizer_path):
            engine = DPRInferenceEngine(query_model, passage_model, tokenizer_path)
            print("✅ DPR Engine loaded successfully in retriever service.")
        else:
            print("⚠️ Warning: Model files not found. Waiting for ingestion step.")

        db_url = os.getenv("DATABASE_URL", "")
        if db_url:
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
            db_engine = create_async_engine(db_url)
            print("✅ Database connection pool initialized.")
        else:
            print("⚠️ Warning: DATABASE_URL not set. /search will be unavailable.")

    except Exception as e:
        print(f"❌ Error during startup: {e}")


@app.get("/health")
def health_check():
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    return {"status": "ok", "service": "retriever"}


@app.post("/embed", response_model=EmbedResponse)
def embed_query(request: EmbedRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized. Ensure models exist.")

    embedding = engine.embed_query(request.text)
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    return {"embedding": embedding}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="DPR engine not initialized.")
    if db_engine is None:
        raise HTTPException(status_code=503, detail="Database not connected. Check DATABASE_URL.")

    # Embed the query
    query_vector = engine.embed_query(request.query)

    # Build optional WHERE filters
    filters = []
    params = {"vec": str(query_vector), "top_k": request.top_k}

    if request.jurisdiction:
        filters.append("jurisdiction = :jurisdiction")
        params["jurisdiction"] = request.jurisdiction
    if request.doc_type:
        filters.append("doc_type = :doc_type")
        params["doc_type"] = request.doc_type

    where_clause = ("WHERE " + " AND ".join(filters)) if filters else ""

    sql = text(f"""
        SELECT
            title,
            source,
            jurisdiction,
            doc_type,
            chunk_index,
            content,
            1 - (embedding <=> CAST(:vec AS vector)) AS score
        FROM documents
        {where_clause}
        ORDER BY embedding <=> CAST(:vec AS vector)
        LIMIT :top_k
    """)

    async with db_engine.connect() as conn:
        rows = (await conn.execute(sql, params)).fetchall()

    results = [
        SearchResult(
            rank=i + 1,
            score=round(float(row[6]), 4),
            title=row[0],
            source=row[1],
            jurisdiction=row[2],
            doc_type=row[3],
            chunk_index=row[4],
            content_preview=row[5][:400] if row[5] else "",
        )
        for i, row in enumerate(rows)
    ]

    return SearchResponse(
        query=request.query,
        total_results=len(results),
        results=results,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)