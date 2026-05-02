import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agent.state import AgentState
from agent.prompts.critic import CRITIC_SYSTEM_PROMPT
from api.schemas.query import StreamEvent

async def critic_node(state: AgentState) -> AgentState:
    """
    Evaluates the draft answer against strict legal standards and triggers retries if necessary.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", CRITIC_SYSTEM_PROMPT),
        ("user", "Original Query: {query}\nDraft Answer: {draft_answer}")
    ])
    
    chain = prompt | llm
    response = await chain.ainvoke({
        "query": state["query"],
        "draft_answer": state.get("draft_answer", "")
    })
    
    try:
        # Parse the critic's JSON evaluation
        evaluation = json.loads(response.content)
        state["critic_scores"] = evaluation.get("scores", {})
        state["overall_score"] = evaluation.get("overall_score", 0.0)
        
        # Decision logic for routing
        overall = state["overall_score"]
        retries = state.get("retry_count", 0)
        
        state["should_retry"] = bool(overall < 0.75 and retries < 2)
        
        state.setdefault("stream_events", []).append(
            StreamEvent(event_type="thought", data=f"Answer evaluated. Confidence score: {overall:.2f}", timestamp=datetime.now().isoformat())
        )
        
        if not state["should_retry"]:
            state["final_answer"] = state["draft_answer"]
            
    except Exception as e:
        import logging
        logging.error(f"Error in critic_node: {e}")
        # Fallback to prevent infinite looping on parse error
        state["should_retry"] = False
        state["final_answer"] = state.get("draft_answer", "Error parsing evaluation.")
        state["overall_score"] = 0.0
        
    return state