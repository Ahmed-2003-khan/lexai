import re
from typing import Dict, List, Any
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Matches === Section Name === lines (with optional surrounding whitespace/newlines)
_HEADER_RE = re.compile(r'===\s*(.+?)\s*===[ \t]*\n?')


class LegalDocumentChunker:
    """Chunks legal documents using semantic boundaries and accurate token counting."""

    def __init__(
        self, 
        chunk_size: int = 400, 
        chunk_overlap: int = 80, 
        tokenizer_name: str = "bert-base-uncased"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # HuggingFace tokenizer for accurate token counting
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._token_count,
            separators=["\n===", "\n---", "\n\n\n", "\n\n", ".\n", ". ", "\n", " "]
        )

    def _token_count(self, text: str) -> int:
        """Calculates the exact token count for a given text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Splits text into chunks, attaches metadata, and filters small chunks."""
        raw_chunks = self.splitter.split_text(text)

        processed_chunks = []
        total_raw = len(raw_chunks)
        doc_title = metadata.get("title", "")
        current_section: str | None = None

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()

            # Extract all === Header === markers from this chunk
            headers = _HEADER_RE.findall(chunk_text)
            # Strip every header line from the content
            clean_content = _HEADER_RE.sub("", chunk_text).strip()

            tokens = self._token_count(clean_content)
            if tokens < 10:
                continue

            if headers:
                current_section = headers[0].strip()
                section_title = f"{doc_title} — {current_section}" if doc_title else current_section
            else:
                # Continuation of the previous section
                if current_section:
                    section_title = (
                        f"{doc_title} — {current_section} (cont.)"
                        if doc_title else f"{current_section} (cont.)"
                    )
                else:
                    section_title = doc_title

            chunk_meta = metadata.copy()
            chunk_meta["title"] = section_title

            processed_chunks.append({
                "text": clean_content,
                "chunk_index": idx,
                "total_chunks": total_raw,
                "token_count": tokens,
                "char_count": len(clean_content),
                "metadata": chunk_meta
            })

        return processed_chunks

    def chunk_document(
        self, 
        loaded_doc: Dict[str, str], 
        title: str, 
        source: str, 
        jurisdiction: str, 
        doc_type: str
    ) -> List[Dict[str, Any]]:
        """Wraps document data into metadata and chunks the content."""
        metadata = {
            "title": title,
            "source": source,
            "jurisdiction": jurisdiction,
            "doc_type": doc_type,
            "filename": loaded_doc.get("filename", "unknown"),
        }
        return self.chunk(loaded_doc["content"], metadata)