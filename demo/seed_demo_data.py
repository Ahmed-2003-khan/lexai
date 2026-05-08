import os
import typer
import asyncio
from pathlib import Path
from typing import Dict, Any

# Ensure the ingestion module can be imported when running from the root directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pipeline import IngestionPipeline

app = typer.Typer(help="CLI tool to seed LexAI database with sample legal data.")


async def async_seed_database(db_url: str):
    """
    Asynchronous function to handle the actual database ingestion process.
    Reads sample text files and pushes them through the ingestion pipeline.
    """
    typer.secho(f"Connecting to database at {db_url}...", fg=typer.colors.CYAN)
    
    # Initialize the ingestion pipeline orchestrator
    pipeline = IngestionPipeline(db_url=db_url)
    
    # Define the directory where the sample data files are located
    sample_dir = Path(__file__).parent / "sample_data"
    
    # Map filenames to their respective legal metadata
    # This ensures each chunk gets the correct title and source for accurate DPR retrieval
    files_to_seed = [
        {
            "path": sample_dir / "ppc_excerpts.txt",
            "title": "Pakistan Penal Code 1860",
            "source": "PPC-1860",
            "jurisdiction": "PK",
            "doc_type": "statute"
        },
        {
            "path": sample_dir / "contract_act_excerpts.txt",
            "title": "Contract Act 1872",
            "source": "CONTRACT-ACT-1872",
            "jurisdiction": "PK",
            "doc_type": "statute"
        },
        {
            "path": sample_dir / "crpc_bail_excerpts.txt",
            "title": "Code of Criminal Procedure — Bail Provisions",
            "source": "CRPC-BAIL",
            "jurisdiction": "PK",
            "doc_type": "statute"
        }
    ]
    
    total_chunks = 0
    total_files = 0
    
    for file_meta in files_to_seed:
        file_path = file_meta["path"]
        
        if file_path.exists():
            typer.echo(f"Processing {file_path.name}...")
            
            # Read the entire text content of the legal document
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Pass the raw text and metadata to the pipeline for chunking and vectorization
            result: Dict[str, Any] = await pipeline.ingest_text(
                text=content,
                title=file_meta["title"],
                source=file_meta["source"],
                jurisdiction=file_meta["jurisdiction"],
                doc_type=file_meta["doc_type"]
            )
            
            chunks_stored = result.get("chunks_stored", 0)
            total_chunks += chunks_stored
            total_files += 1
            
            typer.echo(f"  -> Ingested {chunks_stored} chunks.")
        else:
            typer.secho(f"File not found: {file_path}. Skipping.", fg=typer.colors.RED)
            
    typer.secho(f"\n✅ Seeded {total_chunks} documents (chunks) from {total_files} files.", fg=typer.colors.GREEN)


@app.command()
def seed(
    db_url: str = typer.Option(
        ..., 
        "--db-url", 
        help="PostgreSQL Database connection URL."
    )
):
    """
    Command line interface entry point to seed the sample data.
    Wraps the async execution in the standard event loop.
    """
    asyncio.run(async_seed_database(db_url))


if __name__ == "__main__":
    app()