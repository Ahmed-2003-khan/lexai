import os
from typing import List
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from tqdm import tqdm


class DPRInferenceEngine:
    """Executes Dense Passage Retrieval inference using optimized ONNX models."""

    def __init__(
        self, 
        query_onnx_path: str, 
        passage_onnx_path: str, 
        tokenizer_path: str, 
        max_length: int = 256
    ):
        """Initializes the ONNX runtime sessions and tokenizer."""
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        providers = ["CPUExecutionProvider"]
        
        self.query_session = ort.InferenceSession(query_onnx_path, providers=providers)
        self.passage_session = ort.InferenceSession(passage_onnx_path, providers=providers)

        query_size_mb = os.path.getsize(query_onnx_path) / (1024 * 1024)
        passage_size_mb = os.path.getsize(passage_onnx_path) / (1024 * 1024)

        print(f"Loaded Query Encoder ONNX ({query_size_mb:.2f} MB)")
        print(f"Loaded Passage Encoder ONNX ({passage_size_mb:.2f} MB)")
        print(f"Loaded Tokenizer from {tokenizer_path}")

    def _tokenize(self, text: str) -> dict:
        """Tokenizes a single text string into numpy arrays required by ONNX."""
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )
        return {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64)
        }

    def _mean_pool_and_normalize(self, last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Applies mean pooling based on attention masks and L2 normalizes the result."""
        input_mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        
        mean_pooled = sum_embeddings / sum_mask
        
        norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normalized = mean_pooled / norms
        
        return normalized[0]

    def embed_query(self, text: str) -> List[float]:
        """Generates a 768-dimensional normalized embedding for a search query."""
        inputs = self._tokenize(text)
        outputs = self.query_session.run(None, inputs)
        
        last_hidden_state = outputs[0]
        embedding = self._mean_pool_and_normalize(last_hidden_state, inputs["attention_mask"])
        
        return embedding.tolist()

    def embed_passage(self, text: str) -> List[float]:
        """Generates a 768-dimensional normalized embedding for a document passage."""
        inputs = self._tokenize(text)
        outputs = self.passage_session.run(None, inputs)
        
        last_hidden_state = outputs[0]
        embedding = self._mean_pool_and_normalize(last_hidden_state, inputs["attention_mask"])
        
        return embedding.tolist()

    def embed_passages_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Processes multiple passages in batches using the ONNX passage encoder."""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding passages"):
            batch_texts = texts[i:i + batch_size]
            
            encoded = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np"
            )
            
            inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64)
            }
            
            outputs = self.passage_session.run(None, inputs)
            last_hidden_states = outputs[0]
            
            input_mask_expanded = np.expand_dims(inputs["attention_mask"], -1)
            sum_embeddings = np.sum(last_hidden_states * input_mask_expanded, axis=1)
            sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
            
            mean_pooled = sum_embeddings / sum_mask
            
            norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-10, norms)
            normalized_batch = mean_pooled / norms
            
            embeddings.extend(normalized_batch.tolist())
            
        return embeddings

    def similarity(self, text1: str, text2: str) -> float:
        """Calculates the dot product similarity score between two queries."""
        emb1 = np.array(self.embed_query(text1))
        emb2 = np.array(self.embed_query(text2))
        return float(np.dot(emb1, emb2))