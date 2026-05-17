import os
from typing import List
from retriever.engine import DPRInferenceEngine

class DocumentEmbedder:
    """
    Generates dense vector embeddings for document chunks. 
    Interfaces exclusively with the custom DPR ONNX Engine.
    """

    def __init__(self, query_encoder_path: str, passage_encoder_path: str, tokenizer_path: str):
        # Limit thread usage to prevent WSL2 CPU/memory exhaustion
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("MKL_NUM_THREADS", "4")
        os.environ.setdefault("ONNXRUNTIME_NUM_THREADS", "4")

        try:
            self.dpr_engine = DPRInferenceEngine(query_encoder_path, passage_encoder_path, tokenizer_path)
            print("✅ DocumentEmbedder initialized with DPR ONNX Engine")
        except Exception as e:
            print(f"⚠️ Error loading DPR engine: {e}")
            self.dpr_engine = None

    @property
    def embedding_dim(self) -> int:
        return 768

    def embed(self, text: str) -> List[float]:
        if not self.dpr_engine:
            raise ValueError("DPR Engine is not initialized.")
        return self.dpr_engine.embed_passage(text)

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not self.dpr_engine:
            raise ValueError("DPR Engine is not initialized.")
        if hasattr(self.dpr_engine, 'embed_passages_batch'):
            return self.dpr_engine.embed_passages_batch(texts, batch_size=batch_size)
        return [self.dpr_engine.embed_passage(text) for text in texts]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_batch(texts)