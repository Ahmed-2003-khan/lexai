import json
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agent.state import AgentState
from agent.prompts.planner import PLANNER_SYSTEM_PROMPT
from api.schemas.query import StreamEvent

async def planner_node(state: AgentState) -> AgentState:
    """
    Breaks down the complex legal query into a series of simpler research tasks.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM_PROMPT),
        ("user", "Query: {query}\nJurisdiction: {jurisdiction}\nDoc Types: {doc_types}")
    ])
    
    chain = prompt | llm
    
    state.setdefault("stream_events", []).append(
        StreamEvent(event_type="thought", data="Analyzing the legal query and creating a research plan...", timestamp=datetime.now().isoformat())
    )
    
    response = await chain.ainvoke({
        "query": state["query"],
        "jurisdiction": state["jurisdiction"],
        "doc_types": ", ".join(state.get("doc_types", [])),
        "conversation_history": state.get("conversation_history", "")
    })
    
    try:
        # Clean up Markdown formatting if the LLM includes it
        raw_content = response.content.strip()
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:]
        elif raw_content.startswith("```"):
            raw_content = raw_content[3:]
            
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]
            
        parsed = json.loads(raw_content.strip())
        
        if isinstance(parsed, dict):
            state["is_relevant"] = parsed.get("is_relevant", True)
            state["plan"] = parsed.get("plan", [])
            if not state["is_relevant"]:
                state["final_answer"] = parsed.get("direct_answer", "Hello! I am LexAI. I can only assist with legal matters. How can I help you today?")
                state["should_retry"] = False
        elif isinstance(parsed, list):
            state["is_relevant"] = True
            state["plan"] = parsed
        else:
            state["is_relevant"] = True
            state["plan"] = [state["query"]]
            
    except Exception as e:
        logging.error(f"Error in planner_node: {e} - Raw Output: {response.content}")
        # Bulletproof Fallback: Assume it's relevant and just use the original query
        state["is_relevant"] = True
        state["plan"] = [state["query"]]
        
    return state