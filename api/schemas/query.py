from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=10, max_length=2000)
    jurisdiction: Optional[str] = Field(default="PK")
    doc_types: Optional[List[str]] = Field(default=["statute", "case_law"])


class SubTask(BaseModel):
    task_id: str
    description: str
    status: Literal["pending", "running", "done", "failed"]


class SearchResult(BaseModel):
    doc_id: str
    title: str
    source: str
    content_snippet: str
    score: float
    citation: str


class QueryResponse(BaseModel):
    query_id: str
    query: str
    answer: str
    citations: List[SearchResult]
    sub_tasks: List[SubTask]
    confidence_score: float
    tokens_used: int
    latency_ms: int


class StreamEvent(BaseModel):
    event_type: Literal["thought", "result", "error"]
    data: str
    timestamp: datetime 