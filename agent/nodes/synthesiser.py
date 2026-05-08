from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agent.state import AgentState
from agent.prompts.synthesiser import SYNTHESISER_SYSTEM_PROMPT
from api.schemas.query import StreamEvent

async def synthesiser_node(state: AgentState) -> AgentState:
    """
    Synthesises raw research database records into a coherent legal answer.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    context_block = ""
    citations_list = []
    
    for idx, res in enumerate(state.get("research_results", [])):
        # Extract the actual legal text returned by the database tool
        outputs = res.get("tool_outputs", [])
        content = ""
        for out in outputs:
            content += str(out.get("output", "")) + "\n"
            
        if not content.strip():
            content = "No documents found for this task."
            
        context_block += f"Task {idx+1} Context:\n{content}\n"
        
        # Build API citations using the real legal text snippet
        if content != "No documents found for this task.":
            citations_list.append({
                "doc_id": f"task_{idx}",
                "title": f"Database Search Result {idx+1}",
                "source": "LexAI Knowledge Base",
                "content_snippet": content[:2000], # Give deepEval ample text to verify
                "score": 1.0,
                "citation": f"Search Task {idx+1}"
            })
            
    state.setdefault("stream_events", []).append(
        StreamEvent(event_type="thought", data=f"Synthesising findings from legal sources...", timestamp=datetime.now().isoformat())
    )
    
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
    state["citations"] = state.get("search_results", citations_list)
    
    return state