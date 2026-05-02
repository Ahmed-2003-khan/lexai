from typing import Optional
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from agent.tools.legal_retriever import LegalVectorRetriever

class CaseLawSearchInput(BaseModel):
    """Input schema for case law searches."""
    query: str = Field(..., description="The legal precedent or case facts to research.")
    jurisdiction: str = Field(default="PK", description="The target jurisdiction code.")

def create_case_law_search_tool(retriever: LegalVectorRetriever) -> StructuredTool:
    """
    Creates a focused search tool exclusively for court decisions and case law.
    """
    async def _search_case_law(query: str, jurisdiction: str = "PK") -> str:
        # Hardcode the doc_types to isolate case precedents
        results = await retriever.search(query=query, jurisdiction=jurisdiction, doc_types=["case_law"])
        
        if not results:
            return "No relevant case law found."
            
        formatted_output = ""
        for res in results:
            # Assuming title holds case name and court/year info for formatting
            formatted_output += f"{res.title} — {res.content}\n\n"
            
        return formatted_output

    return StructuredTool.from_function(
        func=None,
        coroutine=_search_case_law,
        name="search_case_law",
        description="Search for court decisions, judgments, and case law precedents.",
        args_schema=CaseLawSearchInput
    )