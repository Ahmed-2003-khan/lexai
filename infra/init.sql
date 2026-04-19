-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents Table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    jurisdiction TEXT NOT NULL,
    doc_type TEXT NOT NULL CHECK (doc_type IN ('statute', 'case_law', 'article')),
    content TEXT NOT NULL,
    embedding vector(768),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- HNSW Index for fast vector similarity search
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

-- Query Logs Table
CREATE TABLE query_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    latency_ms INT NOT NULL,
    tokens_used INT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);