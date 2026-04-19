# LexAI Legal Research API

A high-performance, robust API built with FastAPI designed for parsing, embedding, and searching legal documents using Retrieval-Augmented Generation (RAG).

## Core Technologies
- **Framework:** FastAPI
- **Database:** PostgreSQL 16 with `pgvector`
- **Cache/Rate Limiting:** Redis
- **AI Tooling:** LangChain, LangGraph, OpenAI
- **Observability:** Prometheus, Grafana, Sentry

## Getting Started

1. Copy the environment file:
   ```bash
   cp .env.example .env