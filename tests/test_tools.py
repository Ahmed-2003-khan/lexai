import pytest
from unittest.mock import AsyncMock, MagicMock
from agent.tools.legal_retriever import LegalVectorRetriever, create_legal_search_tool
from agent.tools.citation_formatter import format_legal_citation
from agent.tools.calculator import legal_calculator

@pytest.mark.asyncio
async def test_legal_retriever_search():
    """Verify that the retriever correctly parses mock database rows into SearchResult models."""
    mock_db_conn = AsyncMock()
    mock_db_conn.fetch.return_value = [
        {"id": "1", "title": "Test Act", "source": "TEST", "content": "Test content", "jurisdiction": "PK", "doc_type": "statute", "score": 0.95}
    ]
    
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768
    
    retriever = LegalVectorRetriever(db_url="mock_url", embedder=mock_embedder)
    
    # Patch the asyncpg connect method directly
    with pytest.MonkeyPatch.context() as m:
        m.setattr("asyncpg.connect", AsyncMock(return_value=mock_db_conn))
        results = await retriever.search("test query")
        
        assert len(results) == 1
        assert results[0].title == "Test Act"
        assert results[0].score == 0.95

def test_citation_formatter():
    """Verify the citation formatter returns the expected string structures."""
    result = format_legal_citation.invoke({"title": "Penal Code", "source": "Act XLV", "doc_type": "statute", "year": "1860"})
    assert "Pakistani Citation Style" in result
    assert "Bluebook Citation Style" in result
    assert "1860" in result

def test_legal_calculator():
    """Verify the calculator regex successfully parses and computes date differences."""
    result = legal_calculator.invoke({"expression": "90 days from 2024-01-15"})
    assert "2024-04-14" in result

def test_tool_definitions():
    """Verify all tool wrappers have correct LangChain properties initialized."""
    mock_retriever = AsyncMock()
    search_tool = create_legal_search_tool(mock_retriever)
    
    assert search_tool.name == "search_legal_documents"
    assert search_tool.description is not None
    assert search_tool.args_schema is not None