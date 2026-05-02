import os
import asyncio
import argparse
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from ingestion.embedder import DocumentEmbedder
from ingestion.pipeline import IngestionPipeline
from retriever.engine import DPRInferenceEngine

load_dotenv()
console = Console()

SEED_FILES = [
    {
        "path": "data/ppc_sections.txt",
        "title": "Pakistan Penal Code 1860",
        "source": "PPC-1860",
        "jurisdiction": "PK",
        "doc_type": "statute"
    },
    {
        "path": "data/contract_act_sections.txt",
        "title": "Contract Act 1872",
        "source": "CONTRACT-ACT-1872",
        "jurisdiction": "PK",
        "doc_type": "statute"
    },
    {
        "path": "data/crpc_bail_sections.txt",
        "title": "Code of Criminal Procedure — Bail Provisions",
        "source": "CRPC-BAIL",
        "jurisdiction": "PK",
        "doc_type": "statute"
    },
    {
        "path": "data/qanun_e_shahadat_sections.txt",
        "title": "Qanun-e-Shahadat Order 1984",
        "source": "QSO-1984",
        "jurisdiction": "PK",
        "doc_type": "statute"
    },
    {
        "path": "data/constitution_fundamental_rights.txt",
        "title": "Constitution of Pakistan 1973 — Fundamental Rights",
        "source": "CONST-1973-PART-II",
        "jurisdiction": "PK",
        "doc_type": "statute"
    },
]

async def test_search(pipeline: IngestionPipeline, embedder: DocumentEmbedder, engine_is_dpr: bool):
    """Executes search tests using the appropriate query encoding strategy."""
    queries = [
        "punishment for murder in Pakistan",
        "requirements for a valid contract",
        "bail conditions for non-bailable offence"
    ]
    
    passed_all = True
    console.print("\n[bold cyan]Running Similarity Search Tests...[/bold cyan]")
    
    async with pipeline.engine.connect() as conn:
        from sqlalchemy import text
        for q in queries:
            
            # Select the appropriate embedding method based on the active engine
            if engine_is_dpr:
                vec = embedder.dpr_engine.embed_query(q)
            else:
                vec = embedder.embed(q)

            # Define the SQL query using CAST for the vector parameter to ensure asyncpg driver compatibility
            query = text("""
                SELECT title, source, 1 - (embedding <=> CAST(:vec AS vector)) AS score 
                FROM documents 
                ORDER BY embedding <=> CAST(:vec AS vector) 
                LIMIT 3
            """)
            
            result = await conn.execute(query, {"vec": str(vec)})
            rows = result.fetchall()
            
            console.print(f"\n[bold]Query:[/bold] '{q}'")
            has_good_match = False
            for row in rows:
                score = row[2]
                console.print(f" - [{score:.4f}] {row[0]} ({row[1]})")
                if score > 0.5:
                    has_good_match = True
                    
            if not has_good_match:
                passed_all = False

    if passed_all:
        console.print("\n[bold green]✅ All test queries passed.[/bold green]")
    else:
        console.print("\n[bold yellow]⚠️ Warning: Some queries returned scores < 0.5[/bold yellow]")

async def main():
    parser = argparse.ArgumentParser(description="Seed LexAI database")
    parser.add_argument("--reset", action="store_true", help="Clear documents table before seeding")
    parser.add_argument("--dry-run", action="store_true", help="Process files without DB insertion")
    parser.add_argument("--use-dpr", action="store_true", help="Use custom trained ONNX DPR models instead of SentenceTransformers")
    args = parser.parse_args()

    db_url = os.getenv("DATABASE_URL")
    if not db_url and not args.dry_run:
        console.print("[red]ERROR: DATABASE_URL not set in environment.[/red]")
        return

    console.print("[bold yellow]Initializing models and pipeline...[/bold yellow]")
    
    dpr_engine = None
    if args.use_dpr:
        console.print("[blue]Loading ONNX DPR Models...[/blue]")
        dpr_engine = DPRInferenceEngine(
            query_onnx_path="models/dpr/query_encoder.onnx",
            passage_onnx_path="models/dpr/passage_encoder.onnx",
            tokenizer_path="models/dpr/tokenizer"
        )
    
    embedder = DocumentEmbedder(dpr_engine=dpr_engine)
    pipeline = IngestionPipeline(db_url=db_url, embedder=embedder) if not args.dry_run else None

    if args.reset and not args.dry_run:
        console.print("[red]Resetting database table...[/red]")
        async with pipeline.engine.begin() as conn:
            from sqlalchemy import text
            await conn.execute(text("TRUNCATE TABLE documents;"))

    summary_data = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Ingesting seed files...", total=len(SEED_FILES))

        for file_meta in SEED_FILES:
            if args.dry_run:
                summary_data.append((file_meta["title"], 10, 300))
                progress.advance(task)
                continue

            try:
                res = await pipeline.ingest_file(
                    path=file_meta["path"],
                    title=file_meta["title"],
                    source=file_meta["source"],
                    jurisdiction=file_meta["jurisdiction"],
                    doc_type=file_meta["doc_type"]
                )
                summary_data.append((file_meta["title"], res["chunks_stored"], res["avg_tokens_per_chunk"]))
            except Exception as e:
                console.print(f"[red]Failed to ingest {file_meta['path']}: {e}[/red]")
            progress.advance(task)

    if not args.dry_run:
        stats = await pipeline.get_stats()
        
    table = Table(title="LexAI Seed Data Summary")
    table.add_column("Document", style="cyan")
    table.add_column("Chunks", justify="right", style="green")
    table.add_column("Avg Tokens", justify="right", style="magenta")

    total_chunks = 0
    total_avg_tokens = 0

    for title, chunks, avg_tok in summary_data:
        table.add_row(title, str(chunks), f"{avg_tok:.1f}")
        total_chunks += chunks
        total_avg_tokens += avg_tok

    if summary_data:
        table.add_section()
        table.add_row("TOTAL", str(total_chunks), f"{(total_avg_tokens / len(summary_data)):.1f}")

    console.print()
    console.print(table)
    console.print("\n[bold green]✅ Vector database is ready for querying.[/bold green]")

    if not args.dry_run:
        await test_search(pipeline, embedder, engine_is_dpr=args.use_dpr)

if __name__ == "__main__":
    asyncio.run(main())