import json
from typing import AsyncGenerator, List
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes.planner import planner_node
from agent.nodes.researcher import create_researcher_node
from agent.nodes.synthesiser import synthesiser_node
from agent.nodes.critic import critic_node
from api.schemas.query import StreamEvent, QueryResponse

def build_legal_agent_graph(tools: list) -> StateGraph:
    """
    Compiles the legal research state machine.
    """
    workflow = StateGraph(AgentState)
    
    # Initialize researcher with tools
    researcher = create_researcher_node(tools)
    
    # Add all functional nodes to the graph
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher)
    workflow.add_node("synthesiser", synthesiser_node)
    workflow.add_node("critic", critic_node)
    
    # Define the sequential execution flow
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "synthesiser")
    workflow.add_edge("synthesiser", "critic")
    
    # Define the conditional edge for the retry loop
    def decide_next_step(state: AgentState) -> str:
        if state.get("should_retry"):
            return "researcher"
        return END
        
    workflow.add_conditional_edges("critic", decide_next_step)
    
    return workflow.compile()

async def run_legal_query(graph, query: str, jurisdiction: str, doc_types: List[str], query_id: str) -> AsyncGenerator[str, None]:
    """
    Executes the compiled graph and yields real-time server-sent events (SSE).
    """
    initial_state = AgentState(
        query=query,
        jurisdiction=jurisdiction,
        doc_types=doc_types,
        query_id=query_id,
        retry_count=0,
        stream_events=[],
        messages=[]
    )
    
    # Stream events from the graph execution
    async for output in graph.astream(initial_state):
        # Find the node that just executed
        for node_name, state_update in output.items():
            # Yield any new stream events appended to the state
            events = state_update.get("stream_events", [])
            if events:
                latest_event = events[-1]
                yield f"data: {latest_event.model_dump_json()}\n\n"
                
            # If the process is finishing, yield the final result
            if node_name == "critic" and not state_update.get("should_retry"):
                final_response = {
                    "answer": state_update.get("final_answer", ""),
                    "citations": state_update.get("citations", []),
                    "confidence_score": state_update.get("overall_score", 0.0)
                }
                # Yield the final compiled answer
                yield f"data: {json.dumps({'event_type': 'result', 'data': json.dumps(final_response)})}\n\n"