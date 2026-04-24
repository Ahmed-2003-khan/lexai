from typing import Dict, List, Any
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
        
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            tokens = self._token_count(chunk_text)
            
            # Filter out chunks that are too small to carry semantic meaning
            if tokens < 10:
                continue
                
            processed_chunks.append({
                "text": chunk_text,
                "chunk_index": idx,
                "total_chunks": total_raw,
                "token_count": tokens,
                "char_count": len(chunk_text),
                "metadata": metadata.copy()
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