import json
import logging
from typing import List, Optional
import asyncpg
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class ConversationService:
    """Manages conversation persistence and memory context building."""

    def __init__(self, db_url: str, redis: Optional[Redis] = None):
        self.db_url = db_url
        self.redis = redis

    async def create_conversation(self, user_id: str, title: Optional[str] = None, jurisdiction: str = "PK") -> dict:
        """Creates a new conversation and returns its metadata."""
        conn = await asyncpg.connect(self.db_url)
        try:
            row = await conn.fetchrow(
                """INSERT INTO conversations (user_id, title, jurisdiction)
                   VALUES ($1, $2, $3)
                   RETURNING id, title, jurisdiction, created_at, updated_at""",
                user_id, title or "New Conversation", jurisdiction
            )
            return dict(row)
        finally:
            await conn.close()

    async def list_conversations(self, user_id: str, limit: int = 30) -> List[dict]:
        """Returns the most recent conversations for a user."""
        conn = await asyncpg.connect(self.db_url)
        try:
            rows = await conn.fetch(
                """SELECT id, title, jurisdiction, created_at, updated_at
                   FROM conversations
                   WHERE user_id = $1
                   ORDER BY updated_at DESC
                   LIMIT $2""",
                user_id, limit
            )
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    async def get_conversation(self, conversation_id: str, user_id: str) -> Optional[dict]:
        """Fetches a conversation with all its messages, verifying ownership."""
        import uuid
        conn = await asyncpg.connect(self.db_url)
        try:
            conv = await conn.fetchrow(
                """SELECT id, title, jurisdiction, created_at, updated_at
                   FROM conversations
                   WHERE id = $1 AND user_id = $2""",
                uuid.UUID(str(conversation_id)), uuid.UUID(str(user_id))
            )
            if not conv:
                return None

            messages = await conn.fetch(
                """SELECT id, role, content, metadata, created_at
                   FROM conversation_messages
                   WHERE conversation_id = $1
                   ORDER BY created_at ASC""",
                uuid.UUID(str(conversation_id))
            )

            parsed_messages = []
            for m in messages:
                msg_dict = dict(m)
                if isinstance(msg_dict.get("metadata"), str):
                    msg_dict["metadata"] = json.loads(msg_dict["metadata"])
                parsed_messages.append(msg_dict)
            
            result = dict(conv)
            result["messages"] = parsed_messages
            return result
        finally:
            await conn.close()

    async def add_message(self, conversation_id: str, role: str, content: str, metadata: dict = None) -> dict:
        """Inserts a message into a conversation and updates the conversation timestamp."""
        import uuid
        conn = await asyncpg.connect(self.db_url)
        try:
            row = await conn.fetchrow(
                """INSERT INTO conversation_messages (conversation_id, role, content, metadata)
                   VALUES ($1, $2, $3, $4)
                   RETURNING id, role, content, metadata, created_at""",
                uuid.UUID(str(conversation_id)), role, content, json.dumps(metadata or {})
            )
            # Update conversation's updated_at timestamp
            await conn.execute(
                "UPDATE conversations SET updated_at = NOW() WHERE id = $1",
                uuid.UUID(str(conversation_id))
            )
            # Invalidate cached context
            if self.redis:
                await self.redis.delete(f"conv:{conversation_id}:context")

            return dict(row)
        finally:
            await conn.close()

    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Deletes a conversation (CASCADE deletes messages). Returns True if deleted."""
        conn = await asyncpg.connect(self.db_url)
        try:
            result = await conn.execute(
                "DELETE FROM conversations WHERE id = $1 AND user_id = $2",
                conversation_id, user_id
            )
            if self.redis:
                await self.redis.delete(f"conv:{conversation_id}:context")
            return result == "DELETE 1"
        finally:
            await conn.close()

    async def build_memory_context(self, conversation_id: str, max_turns: int = 5) -> str:
        """
        Builds a formatted string of the last N Q&A turns for prompt injection.
        Uses Redis cache when available.
        """
        import uuid
        # Check cache first
        if self.redis:
            cached = await self.redis.get(f"conv:{conversation_id}:context")
            if cached:
                return cached

        conn = await asyncpg.connect(self.db_url)
        try:
            # Fetch last max_turns * 2 messages (each turn = user + assistant)
            rows = await conn.fetch(
                """SELECT role, content FROM conversation_messages
                   WHERE conversation_id = $1
                   ORDER BY created_at DESC
                   LIMIT $2""",
                uuid.UUID(str(conversation_id)), max_turns * 2
            )
        finally:
            await conn.close()

        if not rows:
            return ""

        # Reverse to chronological order
        rows = list(reversed(rows))

        # Format as readable conversation history
        lines = []
        for row in rows:
            prefix = "User" if row["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {row['content']}")

        context = "\n".join(lines)

        # Cache for 5 minutes
        if self.redis:
            await self.redis.set(f"conv:{conversation_id}:context", context, ex=300)

        return context

    async def auto_title(self, conversation_id: str, first_query: str) -> None:
        """Sets the conversation title based on the first query (truncated to 50 chars)."""
        import uuid
        title = first_query[:50].strip()
        if len(first_query) > 50:
            title += "..."

        conn = await asyncpg.connect(self.db_url)
        try:
            await conn.execute(
                "UPDATE conversations SET title = $1 WHERE id = $2",
                title, uuid.UUID(str(conversation_id))
            )
        finally:
            await conn.close()
