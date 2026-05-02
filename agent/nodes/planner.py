import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from agent.state import AgentState
from agent.prompts.planner import planner_prompt
from api.schemas.query import StreamEvent

async def planner_node(state: AgentState) -> AgentState:
    """
    Generates a structured research plan based on the user's legal query.
    """
    # Initialize the LLM with strict temperature for deterministic planning
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = planner_prompt | llm
    
    # Notify the user via stream event
    state.setdefault("stream_events", []).append(
        StreamEvent(event_type="thought", data="Breaking down your question into research tasks...", timestamp=datetime.now().isoformat())
    )
    
    try:
        # Execute the chain to get the plan
        response = await chain.ainvoke({
            "query": state["query"],
            "jurisdiction": state["jurisdiction"]
        })
        
        # Parse the JSON array of sub-tasks
        plan = json.loads(response.content)
        state["plan"] = plan
    except Exception as e:
        # Fallback to the original query if JSON parsing fails
        state["plan"] = [state["query"]]
        
    return state