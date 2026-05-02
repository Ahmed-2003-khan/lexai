import logging
import asyncpg
from typing import List, Optional, Type, Dict, Any
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from retriever.engine import DPRInferenceEngine
from api.schemas.query import SearchResult

# Initialize logger for the retrieval tool
logger = logging.getLogger(__name__)

class LegalVectorRetriever:
    """
    Handles connection and semantic vector search against the pgvector database.
    """
    def __init__(self, db_url: str, embedder: DPRInferenceEngine, top_k: int = 10):
        self.db_url = db_url
        self.embedder = embedder
        self.top_k = top_k

    async def search(
        self, 
        query: str, 
        jurisdiction: Optional[str] = None, 
        doc_types: Optional[List[str]] = None, 
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Executes a cosine similarity search in the database using the query vector.
        """
        limit = top_k or self.top_k
        
        # Convert the incoming text query into a dense vector
        query_vector = self.embedder.embed_query(query)

        # Standard SQL query utilizing pgvector's cosine distance operator (<=>)
        sql = """
            SELECT id, title, source, content, jurisdiction, doc_type, 
                   1 - (embedding <=> CAST($1 AS vector)) as score
            FROM documents
            WHERE ($2::text IS NULL OR jurisdiction = $2)
              AND ($3::text[] IS NULL OR doc_type = ANY($3))
            ORDER BY embedding <=> CAST($1 AS vector)
            LIMIT $4
        """
        
        # Establish connection and execute query
        conn = await asyncpg.connect(self.db_url)
        try:
            rows = await conn.fetch(sql, str(query_vector), jurisdiction, doc_types, limit)
            
            results = []
            for row in rows:
                result_dict = dict(row)
                results.append(SearchResult(**result_dict))
                
            if results:
                logger.debug(f"Query: '{query}' | Top Match Score: {results[0].score:.4f}")
                
            return results
        finally:
            await conn.close()


class LegalSearchInput(BaseModel):
    """Input schema for the general legal search tool."""
    query: str = Field(..., description="The legal question or search terms.")
    jurisdiction: str = Field(default="PK", description="The target jurisdiction code.")
    doc_types: List[str] = Field(default=["statute", "case_law"], description="Types of documents to search.")


def create_legal_search_tool(retriever: LegalVectorRetriever) -> StructuredTool:
    """
    Factory function to create the LangChain tool wrapper for the retriever.
    """
    async def _search_func(query: str, jurisdiction: str = "PK", doc_types: List[str] = ["statute", "case_law"]) -> str:
        results = await retriever.search(query=query, jurisdiction=jurisdiction, doc_types=doc_types)
        
        if not results:
            return "No relevant legal documents found for this query."
            
        # Format the retrieved results into a readable string for the LLM
        formatted_output = ""
        for i, res in enumerate(results, 1):
            formatted_output += f"--- Result {i} ---\n"
            formatted_output += f"Title: {res.title}\nSource: {res.source}\nScore: {res.score:.4f}\n"
            formatted_output += f"Content Snippet:\n{res.content}\n\n"
            
        return formatted_output

    return StructuredTool.from_function(
        func=None,
        coroutine=_search_func,
        name="search_legal_documents",
        description="Search the legal document database for statutes, case law, and legal commentary relevant to the query. Use this for any question about law, legal procedure, rights, or legal definitions.",
        args_schema=LegalSearchInput
    )