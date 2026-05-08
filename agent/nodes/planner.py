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
        "doc_types": ", ".join(state.get("doc_types", []))
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
            
        plan = json.loads(raw_content.strip())
        
        # Ensure it's a list
        if isinstance(plan, dict) and "plan" in plan:
            plan = plan["plan"]
        elif not isinstance(plan, list):
            plan = [state["query"]]
            
        state["plan"] = plan
        
    except Exception as e:
        logging.error(f"Error in planner_node: {e} - Raw Output: {response.content}")
        # Bulletproof Fallback: If JSON parsing completely fails, just use the original query as the only task
        state["plan"] = [state["query"]]
        
    return state