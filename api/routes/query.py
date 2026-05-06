from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from api.schemas.query import QueryRequest, QueryResponse
from api.services.query_service import QueryService
from api.routes.auth import get_current_user
from api.config import get_settings

settings = get_settings()

# Make sure tools and graph are initialized properly here or in main.py
# For brevity, assuming `legal_agent_graph` is imported or built globally here
from agent.graph import build_legal_agent_graph
legal_agent_graph = build_legal_agent_graph([]) # Provide actual tools here in production
query_service = QueryService(settings.DATABASE_URL, legal_agent_graph)

router = APIRouter(prefix="/api/v1", tags=["query"])

@router.post("/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest, user: dict = Depends(get_current_user)):
    return await query_service.execute_query(request, user["user_id"])

@router.get("/stream")
async def stream_query(query: str, jurisdiction: str = "PK", doc_types: list[str] = Query(["statute", "case_law"]), user: dict = Depends(get_current_user)):
    request = QueryRequest(query=query, jurisdiction=jurisdiction, doc_types=doc_types)
    return StreamingResponse(query_service.stream_query(request, user["user_id"]), media_type="text/event-stream")

@router.get("/history")
async def get_history(limit: int = 20, user: dict = Depends(get_current_user)):
    return await query_service.get_query_history(user["user_id"], limit)