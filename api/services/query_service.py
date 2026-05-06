import json
import uuid
import asyncpg
from typing import AsyncGenerator, List
from api.schemas.query import QueryRequest, QueryResponse
from agent.graph import run_legal_query

class QueryService:
    def __init__(self, db_url: str, agent_graph):
        self.db_url = db_url
        self.agent_graph = agent_graph

    async def _log_query(self, user_id: str, query: str, jurisdiction: str, doc_types: List[str], response: dict):
        conn = await asyncpg.connect(self.db_url)
        try:
            await conn.execute(
                "INSERT INTO query_logs (user_id, query, jurisdiction, doc_types, response) VALUES ($1, $2, $3, $4, $5)",
                user_id, query, jurisdiction, doc_types, json.dumps(response)
            )
        finally:
            await conn.close()

    async def execute_query(self, request: QueryRequest, user_id: str) -> QueryResponse:
        query_id = str(uuid.uuid4())
        final_response = None
        
        async for event in run_legal_query(self.agent_graph, request.query, request.jurisdiction, request.doc_types, query_id):
            if event.startswith("data: "):
                try:
                    data = json.loads(event.replace("data: ", "").strip())
                    if data.get("event_type") == "result":
                        final_response = json.loads(data["data"])
                except:
                    pass
                    
        if final_response:
            await self._log_query(user_id, request.query, request.jurisdiction, request.doc_types, final_response)
            return QueryResponse(
                query_id=query_id, query=request.query, answer=final_response.get("answer", ""),
                citations=final_response.get("citations", []), sub_tasks=[],
                confidence_score=final_response.get("confidence_score", 0.0), tokens_used=0, latency_ms=0
            )
        raise ValueError("Failed to generate response")

    async def stream_query(self, request: QueryRequest, user_id: str) -> AsyncGenerator[str, None]:
        query_id = str(uuid.uuid4())
        async for event in run_legal_query(self.agent_graph, request.query, request.jurisdiction, request.doc_types, query_id):
            yield event
            if "event_type" in event and "result" in event:
                try:
                    data = json.loads(event.replace("data: ", "").strip())
                    await self._log_query(user_id, request.query, request.jurisdiction, request.doc_types, json.loads(data["data"]))
                except:
                    pass

    async def get_query_history(self, user_id: str, limit: int = 20) -> List[dict]:
        conn = await asyncpg.connect(self.db_url)
        try:
            rows = await conn.fetch("SELECT id, query, jurisdiction, response, created_at FROM query_logs WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2", user_id, limit)
            return [dict(row) for row in rows]
        finally:
            await conn.close()