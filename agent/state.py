from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from api.schemas.query import SearchResult, StreamEvent

class AgentState(TypedDict):
    """
    Represents the state of the legal research agent throughout the graph execution.
    """
    query: str
    jurisdiction: str
    doc_types: List[str]
    plan: List[str]
    research_results: List[dict]
    search_results: List[SearchResult]
    draft_answer: str
    final_answer: str
    citations: List[SearchResult]
    critic_scores: Dict[str, float]
    overall_score: float
    should_retry: bool
    retry_count: int
    messages: List[BaseMessage]
    query_id: str
    error: Optional[str]
    stream_events: List[StreamEvent]