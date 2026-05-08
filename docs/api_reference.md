# LexAI API Reference

This document outlines the REST API endpoints available in LexAI. The API is built with FastAPI and follows standard OpenAPI specifications.
**Base URL:** `http://localhost:8000` (Local) / `https://api.yourdomain.com` (Production)

---

## 1. System Endpoints

### 1.1. Liveness Check
Checks if the API server is running.
* **URL:** `/health`
* **Method:** `GET`
* **Auth Required:** No

**Success Response:**
* **Code:** 200 OK
* **Content:** `{"status": "ok"}`

### 1.2. Readiness Check
Checks if the API is connected to the PostgreSQL database and is ready to serve requests.
* **URL:** `/health/ready`
* **Method:** `GET`
* **Auth Required:** No

**Success Response:**
* **Code:** 200 OK
* **Content:** `{"status": "ready", "database": "connected"}`

**Error Response:**
* **Code:** 503 Service Unavailable (If DB is down)

---

## 2. Authentication

### 2.1. Obtain Access Token
Generates a JWT access token valid for 30 minutes.
* **URL:** `/auth/token`
* **Method:** `POST`
* **Auth Required:** No
* **Content-Type:** `application/x-www-form-urlencoded` or `application/json`

**Request Body:**
```json
{
  "username": "ahmed_admin",
  "password": "supersecretpassword"
}
```

**Success Response:**
* **Code:** 200 OK
* **Content:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5c...",
  "token_type": "bearer"
}
```

**Error Responses:**
* **Code:** 401 Unauthorized (Invalid credentials)

---

## 3. Legal Research (RAG)

### 3.1. Execute Query
Runs a legal query through the LangGraph agent, searches the PGVector database, and synthesizes an answer with citations.
* **URL:** `/api/v1/query`
* **Method:** `POST`
* **Auth Required:** Yes (Bearer Token)
* **Content-Type:** `application/json`

**Request Body:**
```json
{
  "query": "What is the punishment for theft in Pakistan?",
  "jurisdiction": "PK",
  "doc_types": ["statute", "case_law"] 
}
```
*(Note: doc_types is optional and defaults to searching all available types).*

**Success Response:**
* **Code:** 200 OK
* **Content:**
```json
{
  "query_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "What is the punishment for theft in Pakistan?",
  "answer": "According to Section 379 of the Pakistan Penal Code, whoever commits theft shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both.",
  "citations": [
    {
      "title": "Pakistan Penal Code",
      "source": "PPC-1860, Section 379",
      "score": 0.89,
      "content_snippet": "Whoever commits theft shall be punished..."
    }
  ],
  "confidence_score": 0.89
}
```

**Error Responses:**
* **Code:** 401 Unauthorized (Missing or invalid token)
* **Code:** 422 Unprocessable Entity (Invalid JSON schema)
* **Code:** 500 Internal Server Error (Agent failure)

**Example cURL:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{"query": "Bail for murder", "jurisdiction": "PK"}'
```

---

## 4. Document Ingestion

### 4.1. Upload Document
Uploads a new legal document, chunks it, generates DPR embeddings, and stores it in the vector database.
* **URL:** `/api/v1/ingest`
* **Method:** `POST`
* **Auth Required:** Yes (Admin Bearer Token)
* **Content-Type:** `multipart/form-data`

**Form Data Parameters:**
* `file`: The document file (.txt, .pdf)
* `title`: "Pakistan Penal Code"
* `source`: "PPC-1860"
* `jurisdiction`: "PK"
* `doc_type`: "statute"

**Success Response:**
* **Code:** 200 OK
* **Content:**
```json
{
  "file": "ppc_excerpts.txt",
  "chunks_created": 150,
  "chunks_stored": 150,
  "duration_seconds": 4.5,
  "avg_tokens_per_chunk": 210.5
}
```
