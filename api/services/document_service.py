import asyncpg
from typing import List, Dict, Any, Optional

class DocumentService:
    def __init__(self, db_url: str, embedder=None):
        self.db_url = db_url
        self.embedder = embedder

    async def get_all(self, page: int = 1, page_size: int = 20, jurisdiction: Optional[str] = None) -> Dict[str, Any]:
        offset = (page - 1) * page_size
        conn = await asyncpg.connect(self.db_url)
        try:
            where_clause = "WHERE jurisdiction = $3" if jurisdiction else ""
            
            # Get total count
            count_query = f"SELECT COUNT(*) FROM documents {where_clause}"
            total = await conn.fetchval(count_query, jurisdiction) if jurisdiction else await conn.fetchval("SELECT COUNT(*) FROM documents")
            
            # Fetch paginated without embeddings
            query = f"""
                SELECT id, title, source, jurisdiction, doc_type, content 
                FROM documents 
                {where_clause}
                ORDER BY created_at DESC LIMIT $1 OFFSET $2
            """
            rows = await conn.fetch(query, page_size, offset, jurisdiction) if jurisdiction else await conn.fetch(query, page_size, offset)
            
            return {
                "items": [dict(row) for row in rows],
                "total": total,
                "page": page,
                "pages": (total + page_size - 1) // page_size
            }
        finally:
            await conn.close()

    async def get_by_id(self, doc_id: str) -> Optional[Dict]:
        conn = await asyncpg.connect(self.db_url)
        try:
            row = await conn.fetchrow("SELECT id, title, source, jurisdiction, doc_type, content FROM documents WHERE id = $1", doc_id)
            return dict(row) if row else None
        finally:
            await conn.close()

    async def delete(self, doc_id: str) -> bool:
        conn = await asyncpg.connect(self.db_url)
        try:
            result = await conn.execute("DELETE FROM documents WHERE id = $1", doc_id)
            return result == "DELETE 1"
        finally:
            await conn.close()

    async def semantic_search(self, query: str, top_k: int = 10, jurisdiction: str = "PK"):
        if not self.embedder:
            raise ValueError("Embedder not configured")
        
        query_vector = self.embedder.embed_query(query)
        conn = await asyncpg.connect(self.db_url)
        try:
            sql = """
                SELECT id, title, source, content, jurisdiction, doc_type, 
                       1 - (embedding <=> CAST($1 AS vector)) as score
                FROM documents
                WHERE ($2::text IS NULL OR jurisdiction = $2)
                ORDER BY embedding <=> CAST($1 AS vector)
                LIMIT $3
            """
            rows = await conn.fetch(sql, str(query_vector), jurisdiction, top_k)
            return [dict(row) for row in rows]
        finally:
            await conn.close()