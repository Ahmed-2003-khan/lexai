from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agent.state import AgentState
from agent.prompts.synthesiser import SYNTHESISER_SYSTEM_PROMPT
from api.schemas.query import StreamEvent

async def synthesiser_node(state: AgentState) -> AgentState:
    """
    Synthesises raw research data into a coherent legal answer.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    # In a full implementation, you extract actual text from the tool executions.
    # For structure, we format the raw research results into a context block.
    context_block = ""
    for idx, res in enumerate(state.get("research_results", [])):
        context_block += f"Task {idx+1} Context: {res.get('tool_calls', 'No calls')}\n"
        
    state.setdefault("stream_events", []).append(
        StreamEvent(event_type="thought", data=f"Synthesising findings from legal sources...", timestamp=datetime.now().isoformat())
    )
    
    # Construct the final generation prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYNTHESISER_SYSTEM_PROMPT),
        ("user", "Question: {query}\nJurisdiction: {jurisdiction}\nContext: {context}")
    ])
    
    chain = prompt | llm
    response = await chain.ainvoke({
        "query": state["query"],
        "jurisdiction": state["jurisdiction"],
        "context": context_block
    })
    
    state["draft_answer"] = response.content
    state["citations"] = state.get("search_results", [])
    
    return state