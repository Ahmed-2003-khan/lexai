from typing import List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from retriever.engine import DPRInferenceEngine


class DocumentEmbedder:
    """Generates dense vector embeddings for document chunks. Interfaces seamlessly with SentenceTransformers or DPR."""

    def __init__(
        self, 
        model_name: str = "sentence-transformers/msmarco-bert-base-dot-v5",
        dpr_engine: Optional[DPRInferenceEngine] = None
    ):
        self.dpr_engine = dpr_engine
        
        if self.dpr_engine:
            print("DocumentEmbedder initialized with custom DPR Engine.")
            return

        self.model_name = model_name
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print(f"DocumentEmbedder initialized with SentenceTransformer on device: {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def embedding_dim(self) -> int:
        """Returns the dimensionality of the generated embeddings."""
        return 768

    def _l2_normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Applies L2 normalization to ensure unit length vectors."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        return vectors / norms

    def embed(self, text: str) -> List[float]:
        """Embeds a single string and returns an L2-normalized float list."""
        if self.dpr_engine:
            return self.dpr_engine.embed_passage(text)

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
        if self.dpr_engine:
            return self.dpr_engine.embed_passages_batch(texts, batch_size=batch_size)

        vectors = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        vectors_norm = self._l2_normalize(vectors)
        return vectors_norm.tolist()