import os
from typing import Dict
import pypdf
import docx
import httpx
import trafilatura


class DocumentLoader:
    """Loads documents from various file formats and URLs."""

    @staticmethod
    def load_pdf(path: str) -> str:
        """Extracts text from a PDF file."""
        text_parts = []
        with open(path, "rb") as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n\n".join(text_parts)

    @staticmethod
    def load_txt(path: str) -> str:
        """Reads a plain text file."""
        with open(path, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def load_docx(path: str) -> str:
        """Extracts text from a DOCX file."""
        doc = docx.Document(path)
        return "\n".join([para.text for para in doc.paragraphs])

    def load(self, path: str) -> Dict[str, str]:
        """Auto-detects file type by extension and loads content."""
        _, ext = os.path.splitext(path)
        ext = ext.lower()

        if ext == ".pdf":
            content = self.load_pdf(path)
        elif ext == ".txt":
            content = self.load_txt(path)
        elif ext == ".docx":
            content = self.load_docx(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        return {
            "content": content,
            "filename": os.path.basename(path),
            "extension": ext
        }

    @staticmethod
    def load_from_url(url: str) -> str:
        """Fetches URL content and extracts main article text."""
        try:
            response = httpx.get(url, timeout=10.0, follow_redirects=True)
            response.raise_for_status()
            content = trafilatura.extract(response.text)
            return content if content else ""
        except Exception:
            return ""