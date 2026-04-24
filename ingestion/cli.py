import os
import asyncio
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

from .pipeline import IngestionPipeline
from .loader import DocumentLoader

load_dotenv()
app = typer.Typer(help="LexAI Document Ingestion CLI")
console = Console()

def get_pipeline() -> IngestionPipeline:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        console.print("[red]ERROR: DATABASE_URL environment variable not set.[/red]")
        raise typer.Exit(code=1)
    return IngestionPipeline(db_url=db_url)

@app.command("ingest-file")
def ingest_file(
    path: str = typer.Option(..., "--path", help="Path to the document file"),
    title: str = typer.Option(..., "--title", help="Title of the document"),
    source: str = typer.Option(..., "--source", help="Source identifier"),
    jurisdiction: str = typer.Option("PK", "--jurisdiction", help="Jurisdiction code"),
    doc_type: str = typer.Option(..., "--doc-type", help="Document type (statute, case_law, article, constitution)")
):
    pipeline = get_pipeline()
    with console.status(f"[bold green]Ingesting {path}..."):
        result = asyncio.run(pipeline.ingest_file(path, title, source, jurisdiction, doc_type))
    
    table = Table(title="Ingestion Result")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    for k, v in result.items():
        table.add_row(k, str(v))
    console.print(table)

@app.command("ingest-directory")
def ingest_directory(
    dir_path: str = typer.Option(..., "--dir", help="Directory to scan recursively"),
    jurisdiction: str = typer.Option("PK", "--jurisdiction", help="Jurisdiction code"),
    doc_type: str = typer.Option("statute", "--doc-type", help="Default document type")
):
    pipeline = get_pipeline()
    files_to_process = []
    
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('.pdf', '.txt', '.docx')):
                files_to_process.append(os.path.join(root, file))
                
    console.print(f"Found {len(files_to_process)} files to ingest.")
    
    for file_path in files_to_process:
        title = os.path.basename(file_path)
        with console.status(f"[bold green]Processing {title}..."):
            result = asyncio.run(pipeline.ingest_file(file_path, title, title, jurisdiction, doc_type))
            console.print(f"✅ {title}: {result['chunks_stored']} chunks stored in {result['duration_seconds']}s")

@app.command("ingest-url")
def ingest_url(
    url: str = typer.Option(..., "--url", help="URL to ingest"),
    title: str = typer.Option(..., "--title", help="Title of the document"),
    source: str = typer.Option(..., "--source", help="Source identifier"),
    jurisdiction: str = typer.Option("PK", "--jurisdiction", help="Jurisdiction code"),
    doc_type: str = typer.Option("article", "--doc-type", help="Document type")
):
    pipeline = get_pipeline()
    with console.status(f"[bold green]Fetching and ingesting {url}..."):
        content = DocumentLoader.load_from_url(url)
        if not content:
            console.print("[red]Failed to extract content from URL.[/red]")
            raise typer.Exit(code=1)
        result = asyncio.run(pipeline.ingest_text(content, title, source, jurisdiction, doc_type))
    
    console.print(f"✅ Successfully ingested URL: {result['chunks_stored']} chunks stored.")

@app.command("stats")
def stats():
    pipeline = get_pipeline()
    with console.status("[bold green]Fetching stats..."):
        db_stats = asyncio.run(pipeline.get_stats())
        
    table = Table(title="Database Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="magenta")
    
    table.add_row("Total Documents", str(db_stats["total_documents"]))
    table.add_row("Total Chunks", str(db_stats["total_chunks"]))
    table.add_section()
    
    for k, v in db_stats["by_jurisdiction"].items():
        table.add_row(f"Jurisdiction: {k}", str(v))
    for k, v in db_stats["by_doc_type"].items():
        table.add_row(f"Type: {k}", str(v))
        
    console.print(table)

@app.command("clear")
def clear(force: bool = typer.Option(False, "--force", help="Skip confirmation")):
    if not force:
        confirm = typer.confirm("Are you sure you want to delete ALL documents?")
        if not confirm:
            raise typer.Abort()
            
    pipeline = get_pipeline()
    async def _clear():
        async with pipeline.engine.begin() as conn:
            from sqlalchemy import text
            await conn.execute(text("TRUNCATE TABLE documents;"))
            
    with console.status("[bold red]Deleting data..."):
        asyncio.run(_clear())
    console.print("✅ All documents deleted.")

if __name__ == "__main__":
    app()