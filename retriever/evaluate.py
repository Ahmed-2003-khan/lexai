import os
import asyncio
from typing import Dict, Any
import typer
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

from retriever.engine import DPRInferenceEngine

load_dotenv()
app = typer.Typer(help="LexAI Retrieval Evaluation CLI")
console = Console()

async def evaluate_retrieval(engine: DPRInferenceEngine, db_url: str) -> Dict[str, Any]:
    """Runs hardcoded queries against the database and evaluates retrieval accuracy."""
    test_queries = [
        "punishment for murder Pakistan",
        "requirements valid contract",
        "bail non-bailable offence",
        "burden of proof court",
        "fundamental rights arrest detention"
    ]
    
    db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
    db_engine = create_async_engine(db_url)
    
    results = []
    passed_count = 0
    total_top_score = 0.0

    async with db_engine.connect() as conn:
        for query_text in test_queries:
            query_vector = engine.embed_query(query_text)
            
            sql = text("""
                SELECT title, source, 1 - (embedding <=> CAST(:vec AS vector)) AS score
                FROM documents
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT 5
            """)
            
            result = await conn.execute(sql, {"vec": str(query_vector)})
            rows = result.fetchall()
            
            top_score = rows[0][2] if rows else 0.0
            passed = top_score > 0.6
            
            if passed:
                passed_count += 1
            total_top_score += top_score
            
            results.append({
                "query": query_text,
                "top_score": top_score,
                "passed": passed,
                "top_result": f"{rows[0][0]} ({rows[0][1]})" if rows else "No results"
            })

    avg_top_score = total_top_score / len(test_queries) if test_queries else 0.0

    table = Table(title="Retrieval Evaluation Results")
    table.add_column("Query", style="cyan", max_width=40)
    table.add_column("Top Match", style="blue", max_width=40)
    table.add_column("Score", justify="right", style="magenta")
    table.add_column("Status", justify="center")

    for res in results:
        status_marker = "[green]PASS[/green]" if res["passed"] else "[red]FAIL[/red]"
        table.add_row(res["query"], res["top_result"], f"{res['top_score']:.4f}", status_marker)

    console.print(table)
    
    summary = f"\n[bold]Overall Result: {passed_count}/{len(test_queries)} passed (Avg Top Score: {avg_top_score:.4f})[/bold]"
    console.print(summary)

    return {
        "queries_tested": len(test_queries),
        "passed": passed_count,
        "avg_top_score": avg_top_score,
        "results": results
    }

@app.command("evaluate")
def evaluate(
    model_dir: str = typer.Option("models/dpr", "--model-dir", help="Directory containing ONNX files"),
):
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        console.print("[red]ERROR: DATABASE_URL environment variable not set.[/red]")
        raise typer.Exit(code=1)

    query_path = os.path.join(model_dir, "query_encoder.onnx")
    passage_path = os.path.join(model_dir, "passage_encoder.onnx")
    tokenizer_path = os.path.join(model_dir, "tokenizer")

    if not all(os.path.exists(p) for p in [query_path, passage_path, tokenizer_path]):
        console.print("[red]ERROR: DPR model files not found. Ensure models/dpr is populated.[/red]")
        raise typer.Exit(code=1)

    with console.status("[bold yellow]Loading DPR Engine...[/bold yellow]"):
        engine = DPRInferenceEngine(query_path, passage_path, tokenizer_path)

    asyncio.run(evaluate_retrieval(engine, db_url))

if __name__ == "__main__":
    app()