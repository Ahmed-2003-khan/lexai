import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import AIMessage
from agent.state import AgentState
from agent.nodes.planner import planner_node
from agent.nodes.critic import critic_node
from agent.graph import build_legal_agent_graph

@pytest.mark.asyncio
async def test_planner_node():
    """Verify the planner node successfully parses JSON arrays from the LLM."""
    state = AgentState(query="test query", jurisdiction="PK")
    
    # Sirf ainvoke method ko mock kar rahay hain
    with patch("agent.nodes.planner.ChatOpenAI.ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        # Asali AIMessage object return karwayen
        mock_ainvoke.return_value = AIMessage(content='["task 1", "task 2"]')
        
        new_state = await planner_node(state)
        assert len(new_state["plan"]) == 2
        assert "task 1" in new_state["plan"]

@pytest.mark.asyncio
async def test_critic_node_retry_logic():
    """Verify the critic node sets should_retry=True when score is below threshold."""
    state = AgentState(query="test", draft_answer="bad answer", retry_count=0)
    
    # Sirf ainvoke method ko mock kar rahay hain
    with patch("agent.nodes.critic.ChatOpenAI.ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value = AIMessage(
            content='{"scores": {"factual_accuracy": 0.5}, "overall_score": 0.6, "should_retry": true}'
        )
        
        new_state = await critic_node(state)
        assert new_state["should_retry"] is True

def test_build_graph():
    """Verify the state machine compiles successfully with valid edges."""
    graph = build_legal_agent_graph([])
    assert graph is not None