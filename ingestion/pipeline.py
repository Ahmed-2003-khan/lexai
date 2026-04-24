import time
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from .loader import DocumentLoader
from .chunker import LegalDocumentChunker
from .embedder import DocumentEmbedder


class IngestionPipeline:
    """Orchestrates loading, chunking, embedding, and database ingestion."""

    def __init__(
        self, 
        db_url: str, 
        embedder: Optional[DocumentEmbedder] = None, 
        chunker: Optional[LegalDocumentChunker] = None
    ):
        self.db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
        self.engine = create_async_engine(self.db_url)
        self.loader = DocumentLoader()
        self.chunker = chunker or LegalDocumentChunker()
        self.embedder = embedder or DocumentEmbedder()

    async def _ensure_schema(self):
        """Ensures the chunk_index and updated_at columns exist (migration)."""
        # Execute raw SQL to ensure required columns exist to avoid schema errors
        async with self.engine.begin() as conn:
            await conn.execute(
                text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunk_index INT DEFAULT 0;")
            )
            await conn.execute(
                text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();")
            )
            await conn.execute(
                text("ALTER TABLE documents DROP CONSTRAINT IF EXISTS docs_source_chunk_unique;")
            )
            await conn.execute(
                text("ALTER TABLE documents ADD CONSTRAINT docs_source_chunk_unique UNIQUE (source, chunk_index);")
            )

    async def _insert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        """Uses asyncpg driver directly for fast batched upsert operations."""
        await self._ensure_schema()
        
        insert_query = """
            INSERT INTO documents 
            (title, source, jurisdiction, doc_type, content, embedding, chunk_index, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6::vector, $7, now(), now())
            ON CONFLICT (source, chunk_index) DO UPDATE 
            SET content = EXCLUDED.content, 
                embedding = EXCLUDED.embedding, 
                updated_at = now()
        """
        
        records = []
        for chunk, emb in zip(chunks, embeddings):
            meta = chunk["metadata"]
            records.append((
                meta["title"],
                meta["source"],
                meta["jurisdiction"],
                meta["doc_type"],
                chunk["text"],
                str(emb),
                chunk["chunk_index"]
            ))

        # Access asyncpg connection directly from SQLAlchemy async engine
        raw_conn = await self.engine.raw_connection()
        try:
            asyncpg_conn = raw_conn.driver_connection
            await asyncpg_conn.executemany(insert_query, records)
        finally:
            await raw_conn.close()

        return len(records)

    async def ingest_file(
        self, path: str, title: str, source: str, jurisdiction: str, doc_type: str
    ) -> Dict[str, Any]:
        """End-to-end ingestion from a file path."""
        start_time = time.time()
        
        loaded_doc = self.loader.load(path)
        chunks = self.chunker.chunk_document(loaded_doc, title, source, jurisdiction, doc_type)
        
        if not chunks:
            return {"file": path, "chunks_created": 0, "chunks_stored": 0, "duration_seconds": 0.0, "avg_tokens_per_chunk": 0.0}
            
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_batch(texts)
        
        stored = await self._insert_chunks(chunks, embeddings)
        
        duration = time.time() - start_time
        avg_tokens = sum(c["token_count"] for c in chunks) / len(chunks)
        
        return {
            "file": path,
            "chunks_created": len(chunks),
            "chunks_stored": stored,
            "duration_seconds": round(duration, 3),
            "avg_tokens_per_chunk": round(avg_tokens, 1)
        }

    async def ingest_text(
        self, text: str, title: str, source: str, jurisdiction: str, doc_type: str
    ) -> Dict[str, Any]:
        """End-to-end ingestion from a raw string."""
        start_time = time.time()
        
        loaded_doc = {"content": text, "filename": "raw_text"}
        chunks = self.chunker.chunk_document(loaded_doc, title, source, jurisdiction, doc_type)
        
        if not chunks:
            return {"file": "raw_text", "chunks_created": 0, "chunks_stored": 0, "duration_seconds": 0.0, "avg_tokens_per_chunk": 0.0}
            
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_batch(texts)
        
        stored = await self._insert_chunks(chunks, embeddings)
        
        duration = time.time() - start_time
        avg_tokens = sum(c["token_count"] for c in chunks) / len(chunks)
        
        return {
            "file": "raw_text",
            "chunks_created": len(chunks),
            "chunks_stored": stored,
            "duration_seconds": round(duration, 3),
            "avg_tokens_per_chunk": round(avg_tokens, 1)
        }

    async def ingest_batch(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ingests a batch of documents sequentially."""
        start_time = time.time()
        results = []
        total_chunks = 0
        
        for item in items:
            res = await self.ingest_text(
                text=item["text"],
                title=item["title"],
                source=item["source"],
                jurisdiction=item["jurisdiction"],
                doc_type=item["doc_type"]
            )
            results.append(res)
            total_chunks += res["chunks_stored"]
            
        return {
            "total_items": len(items),
            "total_chunks": total_chunks,
            "total_duration_seconds": round(time.time() - start_time, 3),
            "results": results
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Fetches aggregation statistics from the database."""
        await self._ensure_schema()
        async with self.engine.connect() as conn:
            from sqlalchemy import text
            total_docs = await conn.scalar(text("SELECT COUNT(DISTINCT source) FROM documents;"))
            total_chunks = await conn.scalar(text("SELECT COUNT(*) FROM documents;"))
            
            by_jurisdiction_rows = await conn.execute(text("SELECT jurisdiction, COUNT(*) FROM documents GROUP BY jurisdiction;"))
            by_jurisdiction = {row[0]: row[1] for row in by_jurisdiction_rows}
            
            by_doc_type_rows = await conn.execute(text("SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type;"))
            by_doc_type = {row[0]: row[1] for row in by_doc_type_rows}
            
        return {
            "total_documents": total_docs or 0,
            "by_jurisdiction": by_jurisdiction,
            "by_doc_type": by_doc_type,
            "total_chunks": total_chunks or 0
        }