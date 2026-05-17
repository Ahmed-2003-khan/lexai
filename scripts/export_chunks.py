"""
Export all chunks from the PostgreSQL documents table to a JSON file.

Usage:
    python scripts/export_chunks.py
    python scripts/export_chunks.py --output data/my_export.json
    python scripts/export_chunks.py --local                           # use localhost instead of Docker host
    python scripts/export_chunks.py --source PPC-1860                 # filter by source
    python scripts/export_chunks.py --db-url postgresql://user:pass@localhost:5432/lexaidb  # full override
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

# ── Root so that ingestion/* imports work ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")
console = Console()

DEFAULT_OUTPUT = ROOT / "data" / "chunks_export.json"


def _build_db_url(local: bool) -> str:
    """
    Build the async SQLAlchemy URL.
    Docker uses hostname 'postgres'; locally we swap it for 'localhost'.
    """
    url = os.getenv("DATABASE_URL", "")
    if not url:
        console.print("[bold red]ERROR:[/bold red] DATABASE_URL not set in .env")
        sys.exit(1)

    # Ensure async driver
    url = url.replace("postgresql://", "postgresql+asyncpg://")

    # When running outside Docker, override the container hostname
    if local:
        url = url.replace("@postgres:", "@localhost:")

    return url


async def fetch_chunks(db_url: str, source_filter: str | None) -> list[dict]:
    """Fetch id, title, content, source, doc_type from documents table."""
    engine = create_async_engine(db_url)

    base_query = """
        SELECT
            id::text        AS id,
            title           AS section_title,
            content,
            source,
            doc_type
        FROM documents
    """

    params: dict = {}
    if source_filter:
        base_query += " WHERE source = :source"
        params["source"] = source_filter

    base_query += " ORDER BY source, chunk_index;"

    async with engine.connect() as conn:
        result = await conn.execute(text(base_query), params)
        rows = result.mappings().all()

    await engine.dispose()

    return [dict(row) for row in rows]


def save_json(chunks: list[dict], output_path: Path) -> None:
    """Write chunks list to a pretty-printed JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Export LexAI chunks to JSON")
    parser.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Connect to localhost instead of Docker 'postgres' host",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Filter by source (e.g. PPC-1860, CONTRACT-ACT-1872)",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        dest="db_url",
        help="Full DB URL override, e.g. postgresql://postgres:secret@localhost:5432/lexaidb",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.db_url:
        db_url = args.db_url.replace("postgresql://", "postgresql+asyncpg://")
    else:
        db_url = _build_db_url(local=args.local)

    console.print("\n[bold cyan]LexAI — Chunk Exporter[/bold cyan]")
    console.print(f"  DB host : [yellow]{'localhost' if args.local else 'postgres (Docker)'}[/yellow]")
    console.print(f"  Filter  : [yellow]{args.source or 'ALL sources'}[/yellow]")
    console.print(f"  Output  : [yellow]{output_path}[/yellow]\n")

    # ── Fetch ──────────────────────────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Fetching chunks from database...", total=None)
        try:
            chunks = await fetch_chunks(db_url, source_filter=args.source)
        except Exception as exc:
            console.print(f"\n[bold red]Connection failed:[/bold red] {exc}")
            console.print(
                "\n[dim]Tip: If running outside Docker, add the [bold]--local[/bold] flag.[/dim]"
            )
            sys.exit(1)
        progress.update(task, completed=1, total=1)

    if not chunks:
        console.print("[bold yellow]⚠  No chunks found. Is the database seeded?[/bold yellow]")
        sys.exit(0)

    # ── Save ───────────────────────────────────────────────────────────────────
    save_json(chunks, output_path)

    # ── Summary table ─────────────────────────────────────────────────────────
    # Count by source
    from collections import Counter
    source_counts: Counter = Counter(c["source"] for c in chunks)

    table = Table(title="Export Summary", show_footer=False)
    table.add_column("Source", style="cyan")
    table.add_column("Chunks", justify="right", style="green")

    for src, count in sorted(source_counts.items()):
        table.add_row(src, str(count))

    table.add_section()
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{len(chunks)}[/bold]")

    console.print()
    console.print(table)
    console.print(
        f"\n[bold green]✅ {len(chunks)} chunks exported →[/bold green] [underline]{output_path}[/underline]\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
