from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agent.state import AgentState
from api.schemas.query import StreamEvent
from agent.prompts.researcher import RESEARCHER_SYSTEM_PROMPT

# Import tools (assuming they are initialized properly in your dependency injection or globally)
# For the sake of the node, we pass the tools array directly. 
# Note: You will inject the instantiated tools when building the graph.

def create_researcher_node(tools: list):
    """
    Factory to create the researcher node with injected tools.
    """
    async def researcher_node(state: AgentState) -> AgentState:
        """
        Executes the research plan by invoking tools for each sub-task.
        """
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        llm_with_tools = llm.bind_tools(tools)
        
        all_research_results = state.get("research_results", [])
        
        # Iterate over each planned sub-task to gather comprehensive data
        for sub_task in state.get("plan", []):
            state.setdefault("stream_events", []).append(
                StreamEvent(event_type="thought", data=f"Researching: {sub_task}", timestamp=datetime.now().isoformat())
            )
            
            # Construct the prompt for the tool-calling LLM
            prompt = f"Research this specific legal question: {sub_task}. Jurisdiction: {state['jurisdiction']}. Search for relevant statutes and case law."
            messages = [
                {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
                HumanMessage(content=prompt)
            ]
            
            # Invoke LLM and allow it to call necessary tools
            response = await llm_with_tools.ainvoke(messages)
            
            # In a production LangGraph, you would use a ToolNode to execute the calls.
            # For this node, we record the raw LLM tool calls and outputs.
            all_research_results.append({
                "sub_task": sub_task,
                "tool_calls": response.tool_calls if hasattr(response, "tool_calls") else []
            })
            
        state["research_results"] = all_research_results
        
        # Increment retry count to prevent infinite loops
        state["retry_count"] = state.get("retry_count", 0) + 1
        return state
        
    return researcher_node