# LexAI ⚖️🤖

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![LangGraph](https://img.shields.io/badge/LangGraph-Latest-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-green)

LexAI is an agentic Retrieval-Augmented Generation (RAG) platform specialized for legal document search and analysis. Powered by a LangGraph state machine and a custom Dense Passage Retrieval (DPR) model, it allows users to ask complex legal questions and receive synthesized answers backed by precise citations from statutes and case law.

## Architecture

```mermaid
graph TD;
    User-->|REST API| FastAPI;
    FastAPI-->|Query| LangGraph_Agent;
    LangGraph_Agent-->|Vector Search| PGVector;
    PGVector-->|Context Chunks| LangGraph_Agent;
    LangGraph_Agent-->|Critic/Retry| PGVector;
    LangGraph_Agent-->|Synthesis| LLM;
    LLM-->|Final Answer & Citations| FastAPI;
    FastAPI-->|JSON| User;
```

## Tech Stack

| Technology | Purpose | Version |
| :--- | :--- | :--- |
| Python | Core Language | 3.11 |
| FastAPI | REST API Framework | 0.109 |
| LangGraph | Agentic State Machine | ^0.0.27 |
| PostgreSQL & PGVector | Vector Database | 16 |
| ONNX Runtime | DPR Embedding Inference | 1.17 |
| Docker | Containerization | Latest |

## Quick Start

### Clone the repo
```bash
git clone https://github.com/yourusername/lexai.git
cd lexai
```

### Setup environment
```bash
cp .env.example .env
# Add your OPENAI_API_KEY and other secrets to .env
```

### Run with Docker
```bash
docker compose up --build -d
```

### Test the API
```bash
curl http://localhost:8000/health/ready

curl -X POST http://localhost:8000/api/v1/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the punishment for theft in Pakistan?", "jurisdiction": "PK"}'
```

## Key API Endpoints

| Method | Path | Description | Example Request Body |
| :--- | :--- | :--- | :--- |
| GET | /health | Liveness probe | None |
| GET | /health/ready | Readiness probe (checks DB) | None |
| POST | /auth/token | Get JWT access token | `{"username": "admin", "password": "..."}` |
| POST | /api/v1/query | Execute legal research | `{"query": "Bail for murder", "jurisdiction": "PK"}` |
| POST | /api/v1/ingest | Upload legal document | Form-data with file |

## Development Setup

To set up the project locally without Docker:

```bash
python -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`
pip install .
pip install -e ".[dev]"

# Run linter
black . && flake8 .

# Run tests
pytest tests/ --ignore=tests/integration
```

## Document Ingestion

### Ingest Custom Documents
Use the CLI script to seed the vector database with your own text documents.
```bash
python scripts/seed_database.py --db-url postgresql://lexai:lexai_password@localhost:5432/lexaidb
```

## Environment Variables

| Variable | Description | Example |
| :--- | :--- | :--- |
| DATABASE_URL | PostgreSQL connection string | postgresql://lexai:pass@postgres:5432/lexaidb |
| OPENAI_API_KEY | OpenAI Key for LLM synthesis | sk-proj-... |
| JWT_SECRET_KEY | Secret for auth token generation | your-256-bit-secret |

## Project Structure

```plaintext
lexai/
├── agent/            # LangGraph workflows and tool definitions
├── api/              # FastAPI routers, schemas, and dependencies
├── data/             # Sample legal text data
├── demo/             # Sample legal text data and minimal UI demo
├── docs/             # Extensive architectural and developer documentation
├── evals/            # LLM evaluation logic and results
├── infra/            # Docker and orchestration configs
├── ingestion/        # Document chunking and metadata injection
├── models/           # ONNX models and local embeddings
├── retriever/        # DPR Inference engine and vector search logic
├── scripts/          # Database seeding and utility scripts
└── tests/            # Unit and integration tests
```