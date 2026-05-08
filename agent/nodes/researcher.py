from datetime import datetime
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agent.state import AgentState
from api.schemas.query import StreamEvent
from agent.prompts.researcher import RESEARCHER_SYSTEM_PROMPT

def create_researcher_node(tools: list):
    """
    Factory to create the researcher node with injected tools.
    """
    tool_map = {tool.name: tool for tool in tools}

    async def researcher_node(state: AgentState) -> AgentState:
        """
        Executes the research plan and physically calls the tools to fetch database records.
        """
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Explicitly bind the exact name of the existing legal search tool
        llm_with_tools = llm.bind_tools(tools, tool_choice="search_legal_documents")
        
        all_research_results = state.get("research_results", [])
        
        for sub_task in state.get("plan", []):
            state.setdefault("stream_events", []).append(
                StreamEvent(event_type="thought", data=f"Researching: {sub_task}", timestamp=datetime.now().isoformat())
            )
            
            prompt = f"Research this specific legal question: {sub_task}. Jurisdiction: {state['jurisdiction']}. You MUST use the search_legal_documents tool to find actual laws."
            messages = [
                {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
                HumanMessage(content=prompt)
            ]
            
            response = await llm_with_tools.ainvoke(messages)
            
            tool_outputs = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                for call in response.tool_calls:
                    tool_name = call["name"]
                    tool_args = call["args"]
                    if tool_name in tool_map:
                        try:
                            # Execute the physical database connection using the provided tool
                            tool_result = await tool_map[tool_name].ainvoke(tool_args)
                            tool_outputs.append({
                                "tool": tool_name,
                                "output": tool_result
                            })
                        except Exception as e:
                            logging.error(f"Error executing tool {tool_name}: {e}")
            
            all_research_results.append({
                "sub_task": sub_task,
                "tool_calls": response.tool_calls if hasattr(response, "tool_calls") else [],
                "tool_outputs": tool_outputs
            })
            
        state["research_results"] = all_research_results
        state["retry_count"] = state.get("retry_count", 0) + 1
        return state
        
    return researcher_node