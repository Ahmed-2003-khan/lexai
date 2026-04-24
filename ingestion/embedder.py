from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class DocumentEmbedder:
    """Generates dense vector embeddings for document chunks."""

    def __init__(self, model_name: str = "sentence-transformers/msmarco-bert-base-dot-v5"):
        self.model_name = model_name
        
        # Auto-detect device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print(f"Initializing DocumentEmbedder on device: {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def embedding_dim(self) -> int:
        return 768

    def _l2_normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Applies L2 normalization to vectors."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-10, norms)
        return vectors / norms

    def embed(self, text: str) -> List[float]:
        """Embeds a single string and returns an L2-normalized float list."""
        vec = self.model.encode([text], convert_to_numpy=True)
        vec_norm = self._l2_normalize(vec)[0]
        return vec_norm.tolist()

    def embed_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32, 
        show_progress: bool = True
    ) -> List[List[float]]:
        """Embeds a list of strings in batches and returns normalized float lists."""
        vectors = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        vectors_norm = self._l2_normalize(vectors)
        return vectors_norm.tolist()