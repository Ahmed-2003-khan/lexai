# Development & Contribution Guide

Welcome to the LexAI development guide! This document will help you set up your local environment, understand our testing protocols, and learn how to extend the system's capabilities.

## 1. Local Environment Setup

To run LexAI locally without Docker (useful for debugging and active development):

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/lexai.git
   cd lexai
   ```

2. **Set up a Python Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install .
   pip install -e ".[dev]"
   ```

4. **Start the Local Database:**
   Use Docker Compose to spin up just the PostgreSQL/PGVector database:
   ```bash
   docker compose up -d postgres
   ```

5. **Environment Variables:**
   Copy `.env.example` to `.env` and fill in your `OPENAI_API_KEY` and local `DATABASE_URL`.

6. **Run the API Server:**
   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

## 2. Testing & Evaluations

LexAI uses a dual-testing approach: standard unit tests for code logic, and LLM evaluations (DeepEval) for RAG accuracy.

**Run Unit & Integration Tests:**
```bash
pytest tests/ --ignore=tests/integration
```
*(Note: Standard unit tests mock the database and LLM calls to run quickly without API costs).*

**Run RAG Evaluations (DeepEval):**
To test the actual retrieval and generation quality against your test cases:
```bash
python evals/run_evals.py
```
*(Warning: This consumes OpenAI API credits. Use the `[:1]` slice in the script to test a single case first).*

## 3. Extending the Agent: Adding a New Tool

LexAI's LangGraph agent can be extended with new tools (e.g., a tool to fetch recent Supreme Court rulings from an external API).

**Create the Tool:** Add a new file in `agent/tools/`.
**Define the Schema:** Use Pydantic to strictly define the expected input parameters.
**Wrap with @tool:**

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class CaseLawInput(BaseModel):
    query: str = Field(description="The legal topic to search for.")

@tool("search_case_law", args_schema=CaseLawInput)
def search_case_law(query: str) -> str:
    """Fetches recent case law from the external API."""
    # Implementation here
    return results
```

**Register the Tool:** Import and add your new tool to the tools list in `agent/graph.py` so the LLM knows it exists.

## 4. Coding Standards

**Code Formatting:** We use `black` for formatting and `flake8` for linting.
```bash
black . && flake8 .
```
