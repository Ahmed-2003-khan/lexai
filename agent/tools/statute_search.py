from typing import Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from agent.tools.legal_retriever import LegalVectorRetriever

class StatuteSearchInput(BaseModel):
    """Input schema for statutory law searches."""
    query: str = Field(..., description="The specific statutory issue to research.")
    jurisdiction: str = Field(default="PK", description="The target jurisdiction code.")

def create_statute_search_tool(retriever: LegalVectorRetriever) -> StructuredTool:
    """
    Creates a focused search tool exclusively for statutory documents.
    """
    async def _search_statutes(query: str, jurisdiction: str = "PK") -> str:
        # Hardcode the doc_types to isolate statutory law
        results = await retriever.search(query=query, jurisdiction=jurisdiction, doc_types=["statute"])
        
        if not results:
            return "No relevant statutes found."
            
        formatted_output = ""
        for res in results:
            formatted_output += f"STATUTE: {res.title}\n"
            formatted_output += f"Source: {res.source}\n"
            formatted_output += f"Relevance: {res.score:.4f}\n\n"
            formatted_output += f"{res.content}\n---\n"
            
        return formatted_output

    return StructuredTool.from_function(
        func=None,
        coroutine=_search_statutes,
        name="search_statutes",
        description="Search specifically for statutory law, legislation, acts of parliament, and legal codes.",
        args_schema=StatuteSearchInput
    )