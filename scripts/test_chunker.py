import os
import sys

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.chunker import LegalDocumentChunker

chunker = LegalDocumentChunker(chunk_size=50, chunk_overlap=10)

text = """=== Section 302: Qatl-i-Amd ===
Whoever commits qatl-i-amd shall, subject to the provisions of this Chapter be punished with death as qisas.

=== Section 303: Qatl committed under ikrah ===
Whoever commits qatl under ikrah-i-tam shall be punished.
This is a very long text that will probably be split into a second chunk because of the small chunk size we set for this test.
"""

metadata = {
    "title": "Pakistan Penal Code",
    "source": "PPC-1860",
    "jurisdiction": "PK",
    "doc_type": "statute"
}

chunks = chunker.chunk(text, metadata)

for c in chunks:
    print(f"\n--- Chunk {c['chunk_index']} of {c['total_chunks']} ---")
    print(f"Text: {c['text']}")
    print(f"Metadata: {c['metadata']}")
