import time
import uuid
import os
import sentry_sdk
from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas.query import QueryRequest, QueryResponse
from api.services.query_service import QueryService
from api.routes.auth import get_current_user
from api.config import get_settings

# Import the custom Prometheus metrics defined for LexAI
from api.metrics import ACTIVE_QUERIES, QUERY_COUNTER, QUERY_LATENCY

# --- NAYE IMPORTS TOOLS AUR RETRIEVER KE LIYE ---
from agent.tools.legal_retriever import LegalVectorRetriever, create_legal_search_tool
from retriever.engine import DPRInferenceEngine
from agent.graph import build_legal_agent_graph

settings = get_settings()

# STEP 1: Embedder (DPR Engine) Setup Karein
try:
    dpr_engine = DPRInferenceEngine(
        os.getenv("QUERY_ENCODER_PATH", "models/dpr/query_encoder.onnx"),
        os.getenv("PASSAGE_ENCODER_PATH", "models/dpr/passage_encoder.onnx"),
        os.getenv("TOKENIZER_PATH", "models/dpr/tokenizer")
    )
except Exception as e:
    import logging
    logging.warning(f"Could not load DPR engine in query router: {e}")
    dpr_engine = None

# STEP 2: Database Retriever aur Tool Setup Karein
db_retriever = LegalVectorRetriever(db_url=settings.DATABASE_URL, embedder=dpr_engine)
search_tool = create_legal_search_tool(db_retriever)

# STEP 3: Graph ko Khali List [] ki bajaye Tool Pass Karein!
legal_agent_graph = build_legal_agent_graph(tools=[search_tool]) 

query_service = QueryService(settings.DATABASE_URL, legal_agent_graph)

router = APIRouter(prefix="/api/v1", tags=["query"])

@router.post("/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest, user: dict = Depends(get_current_user)):
    """Processes a legal query and tracks observability metrics."""
    query_id = str(uuid.uuid4())
    jurisdiction = request.jurisdiction if request.jurisdiction else "Pakistan"
    user_id = user.get("user_id", "unknown")

    # Attach context to Sentry for error tracing
    sentry_sdk.set_tag("query_id", query_id)
    sentry_sdk.set_tag("user_id", user_id)

    # Track real-time active queries
    ACTIVE_QUERIES.inc()
    start_time = time.time()
    
    try:
        response = await query_service.execute_query(request, user_id, query_id)
        
        # Increment success counter
        QUERY_COUNTER.labels(jurisdiction=jurisdiction, status="success").inc()
        
        return response
        
    except Exception as e:
        # Increment failure counter
        QUERY_COUNTER.labels(jurisdiction=jurisdiction, status="failed").inc()
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Decrement active queries and record the execution latency
        ACTIVE_QUERIES.dec()
        latency = time.time() - start_time
        QUERY_LATENCY.observe(latency)

@router.get("/stream")
async def stream_query(query: str, jurisdiction: str = "PK", doc_types: list[str] = Query(["statute", "case_law"]), user: dict = Depends(get_current_user)):
    """Streams the response for a legal query."""
    query_id = str(uuid.uuid4())
    request = QueryRequest(query=query, jurisdiction=jurisdiction, doc_types=doc_types)
    return StreamingResponse(query_service.stream_query(request, user["user_id"], query_id), media_type="text/event-stream")

@router.get("/history")
async def get_history(limit: int = 20, user: dict = Depends(get_current_user)):
    """Retrieves the query history for the authenticated user."""
    return await query_service.get_query_history(user["user_id"], limit)