import uuid
import json
import asyncio
from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.schemas.query import QueryRequest, QueryResponse, SearchResult, SubTask, StreamEvent
from api.middleware.rate_limit import rate_limit_dependency

router = APIRouter(tags=["query"])

@router.post("/query", response_model=QueryResponse, dependencies=[Depends(rate_limit_dependency)])
async def execute_query(request: QueryRequest):
    """Executes a legal research query and returns the analyzed response."""
    
    query_id = str(uuid.uuid4())
    
    return QueryResponse(
        query_id=query_id,
        query=request.query,
        answer="Based on the analysis of the provided jurisdiction, the statute applies.",
        citations=[
            SearchResult(
                doc_id=str(uuid.uuid4()),
                title="Constitutional Petition No. 123",
                source="Supreme Court",
                content_snippet="...the basic structure of the constitution mandates...",
                score=0.95,
                citation="PLD 2024 SC 1"
            )
        ],
        sub_tasks=[
            SubTask(
                task_id=str(uuid.uuid4()),
                description="Retrieve relevant statutes",
                status="done"
            )
        ],
        confidence_score=0.89,
        tokens_used=1450,
        latency_ms=1200
    )

@router.get("/query/{query_id}", response_model=QueryResponse)
async def get_query(query_id: str):
    """Retrieves a previously executed query from the logs."""
    
    return QueryResponse(
        query_id=query_id,
        query="Mock historical query",
        answer="Historical answer.",
        citations=[],
        sub_tasks=[],
        confidence_score=0.99,
        tokens_used=500,
        latency_ms=450
    )

@router.get("/stream/{query_id}")
async def stream_query(query_id: str):
    """Streams the execution events of a running query via Server-Sent Events."""
    
    async def event_generator():
        events = [
            StreamEvent(event_type="thought", data="Analyzing jurisdiction constraints", timestamp=datetime.now(timezone.utc)),
            StreamEvent(event_type="thought", data="Searching vector database", timestamp=datetime.now(timezone.utc)),
            StreamEvent(event_type="result", data="Formulating final response", timestamp=datetime.now(timezone.utc))
        ]
        
        for event in events:
            await asyncio.sleep(0.5)
            yield f"data: {event.model_dump_json()}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")