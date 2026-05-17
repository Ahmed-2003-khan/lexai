from datetime import datetime
from typing import List, Optional
import uuid
from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    """Request body to start a new conversation."""
    title: Optional[str] = Field(default=None, max_length=200)
    jurisdiction: Optional[str] = Field(default="PK")


class ConversationResponse(BaseModel):
    """Single conversation metadata for listing."""
    id: uuid.UUID
    title: str
    jurisdiction: str
    created_at: datetime
    updated_at: datetime


class ConversationMessage(BaseModel):
    """Single message within a conversation."""
    id: uuid.UUID
    role: str
    content: str
    metadata: dict = {}
    created_at: datetime


class ConversationDetail(BaseModel):
    """Full conversation with all messages."""
    id: uuid.UUID
    title: str
    jurisdiction: str
    messages: List[ConversationMessage]
    created_at: datetime
    updated_at: datetime


class MemoryQueryRequest(BaseModel):
    """Query request scoped to a conversation for memory context."""
    query: str = Field(min_length=2, max_length=2000)
    doc_types: Optional[List[str]] = Field(default=["statute", "case_law"])
