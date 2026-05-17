import uuid
import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.schemas.conversation import ConversationCreate, ConversationResponse, ConversationDetail, MemoryQueryRequest
from api.services.conversation_service import ConversationService
from api.schemas.query import QueryRequest
from api.routes.auth import get_current_user
from api.config import get_settings

# Need access to the query service and agent graph initialized in main/query routes
# To avoid circular imports, we'll fetch the query service from the query router or initialize a new one.
# For simplicity, we can just import the already configured query_service from api.routes.query
from api.routes.query import query_service

settings = get_settings()
router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])

# Dependency to get conversation service. Note we don't have redis easily accessible here without app.state
# but we can initialize ConversationService with just db_url for now, or fetch from request.app.state.
def get_conversation_service():
    # In a real app, you might want to pass redis from app state
    return ConversationService(db_url=settings.DATABASE_URL)


@router.post("", response_model=ConversationResponse)
async def create_conversation(
    request: ConversationCreate,
    user: dict = Depends(get_current_user),
    conv_service: ConversationService = Depends(get_conversation_service)
):
    """Start a new conversation."""
    return await conv_service.create_conversation(
        user_id=user["user_id"],
        title=request.title,
        jurisdiction=request.jurisdiction
    )


@router.get("", response_model=List[ConversationResponse])
async def list_conversations(
    limit: int = Query(30, ge=1, le=100),
    user: dict = Depends(get_current_user),
    conv_service: ConversationService = Depends(get_conversation_service)
):
    """List recent conversations for the current user."""
    return await conv_service.list_conversations(user_id=user["user_id"], limit=limit)


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    user: dict = Depends(get_current_user),
    conv_service: ConversationService = Depends(get_conversation_service)
):
    """Get a conversation and all its messages."""
    conv = await conv_service.get_conversation(conversation_id, user["user_id"])
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: dict = Depends(get_current_user),
    conv_service: ConversationService = Depends(get_conversation_service)
):
    """Delete a conversation."""
    success = await conv_service.delete_conversation(conversation_id, user["user_id"])
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found or cannot be deleted")
    return {"message": "Deleted successfully"}


@router.post("/{conversation_id}/query")
async def execute_memory_query(
    conversation_id: str,
    request: MemoryQueryRequest,
    user: dict = Depends(get_current_user),
    conv_service: ConversationService = Depends(get_conversation_service)
):
    """Execute a query within a conversation context synchronously."""
    # Verify conversation exists
    conv = await conv_service.get_conversation(conversation_id, user["user_id"])
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Save user message
    await conv_service.add_message(conversation_id, role="user", content=request.query)

    # Auto-title if it's the first message
    if len(conv.get("messages", [])) == 0:
        await conv_service.auto_title(conversation_id, request.query)

    # Build context
    history = await conv_service.build_memory_context(conversation_id)

    # Execute
    query_id = str(uuid.uuid4())
    qr = QueryRequest(query=request.query, jurisdiction=conv["jurisdiction"], doc_types=request.doc_types)
    
    response = await query_service.execute_query(qr, user["user_id"], query_id, conversation_history=history)

    # Save assistant message
    await conv_service.add_message(
        conversation_id, 
        role="assistant", 
        content=response.answer,
        metadata={"citations": [c.model_dump() for c in response.citations], "confidence_score": response.confidence_score}
    )

    return response


@router.get("/{conversation_id}/stream")
async def stream_memory_query(
    conversation_id: str,
    query: str,
    doc_types: list[str] = Query(["statute", "case_law"]),
    user: dict = Depends(get_current_user),
    conv_service: ConversationService = Depends(get_conversation_service)
):
    """Stream a query within a conversation context."""
    # Verify conversation exists
    conv = await conv_service.get_conversation(conversation_id, user["user_id"])
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Save user message
    await conv_service.add_message(conversation_id, role="user", content=query)

    # Auto-title if it's the first message
    if len(conv.get("messages", [])) == 0:
        await conv_service.auto_title(conversation_id, query)

    # Build context
    history = await conv_service.build_memory_context(conversation_id)

    query_id = str(uuid.uuid4())
    qr = QueryRequest(query=query, jurisdiction=conv["jurisdiction"], doc_types=doc_types)

    async def stream_and_save():
        final_answer = ""
        final_metadata = {}
        
        async for event in query_service.stream_query(qr, user["user_id"], query_id, conversation_history=history):
            yield event
            
            # Intercept final result to save assistant message
            if "event_type" in event and "result" in event:
                try:
                    data_str = event.replace("data: ", "").strip()
                    data = json.loads(data_str)
                    result_data = json.loads(data["data"])
                    final_answer = result_data.get("answer", "")
                    final_metadata = {
                        "citations": result_data.get("citations", []),
                        "confidence_score": result_data.get("confidence_score", 0.0)
                    }
                except Exception as e:
                    import logging
                    logging.error(f"Error parsing final stream result to save message: {e}")

        # Save assistant message after stream completes
        if final_answer:
            await conv_service.add_message(
                conversation_id, 
                role="assistant", 
                content=final_answer,
                metadata=final_metadata
            )

    return StreamingResponse(stream_and_save(), media_type="text/event-stream")
