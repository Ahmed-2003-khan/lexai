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
        """Stores paths and tokenizer. ONNX sessions are loaded lazily on first use."""
        self.max_length = max_length
        self._query_onnx_path = query_onnx_path
        self._passage_onnx_path = passage_onnx_path

        # Limit thread usage to prevent WSL2 CPU/memory exhaustion
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("MKL_NUM_THREADS", "4")
        os.environ.setdefault("ONNXRUNTIME_NUM_THREADS", "4")

        self._sess_options = ort.SessionOptions()
        self._sess_options.intra_op_num_threads = 4
        self._sess_options.inter_op_num_threads = 4

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Sessions are None until first use
        self._query_session: ort.InferenceSession | None = None
        self._passage_session: ort.InferenceSession | None = None

        print(f"Loaded Tokenizer from {tokenizer_path}")
        print("ONNX sessions will be loaded lazily on first use.")

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    def _get_query_session(self) -> ort.InferenceSession:
        """Returns the query encoder session, loading it from disk if not yet loaded."""
        if self._query_session is None:
            size_mb = os.path.getsize(self._query_onnx_path) / (1024 * 1024)
            print(f"[DPR] Loading Query Encoder ONNX ({size_mb:.2f} MB)...")
            self._query_session = ort.InferenceSession(
                self._query_onnx_path,
                sess_options=self._sess_options,
                providers=["CPUExecutionProvider"]
            )
        return self._query_session

    def _get_passage_session(self) -> ort.InferenceSession:
        """Returns the passage encoder session, loading it from disk if not yet loaded."""
        if self._passage_session is None:
            size_mb = os.path.getsize(self._passage_onnx_path) / (1024 * 1024)
            print(f"[DPR] Loading Passage Encoder ONNX ({size_mb:.2f} MB)...")
            self._passage_session = ort.InferenceSession(
                self._passage_onnx_path,
                sess_options=self._sess_options,
                providers=["CPUExecutionProvider"]
            )
        return self._passage_session

    # ── Unloaders ─────────────────────────────────────────────────────────────

    def unload_query_session(self) -> None:
        """Deletes the query encoder session and frees its memory."""
        if self._query_session is not None:
            del self._query_session
            self._query_session = None
            print("[DPR] Query encoder unloaded from memory.")

    def unload_passage_session(self) -> None:
        """Deletes the passage encoder session and frees its memory."""
        if self._passage_session is not None:
            del self._passage_session
            self._passage_session = None
            print("[DPR] Passage encoder unloaded from memory.")

    def unload_all(self) -> None:
        """Unloads both encoder sessions."""
        self.unload_query_session()
        self.unload_passage_session()

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
        outputs = self._get_query_session().run(None, inputs)

        last_hidden_state = outputs[0]
        embedding = self._mean_pool_and_normalize(last_hidden_state, inputs["attention_mask"])

        return embedding.tolist()

    def embed_passage(self, text: str) -> List[float]:
        """Generates a 768-dimensional normalized embedding for a document passage."""
        inputs = self._tokenize(text)
        outputs = self._get_passage_session().run(None, inputs)

        last_hidden_state = outputs[0]
        embedding = self._mean_pool_and_normalize(last_hidden_state, inputs["attention_mask"])

        return embedding.tolist()

    def embed_passages_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Processes multiple passages to generate embeddings using the ONNX passage encoder.
        Executes inference sequentially to align with the fixed ONNX graph dimensions.
        """
        embeddings = []
        
        # Iterate over each text individually to satisfy the model's single-sequence input requirement
        for text in tqdm(texts, desc="Embedding passages"):
            
            # Tokenize a single text string into numpy arrays
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np"
            )
            
            # Prepare the input dictionary for the ONNX runtime session
            inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64)
            }
            
            # Execute the ONNX computation graph
            outputs = self._get_passage_session().run(None, inputs)
            
            # Extract the hidden states from the model output
            last_hidden_states = outputs[0]
            
            # Expand the attention mask to match the hidden state dimensions for broadcasting
            input_mask_expanded = np.expand_dims(inputs["attention_mask"], -1)
            
            # Compute the sum of embeddings, zeroing out padding tokens via the mask
            sum_embeddings = np.sum(last_hidden_states * input_mask_expanded, axis=1)
            
            # Calculate the number of valid tokens to use as the denominator
            sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
            
            # Perform mean pooling to get a single vector representation for the passage
            mean_pooled = sum_embeddings / sum_mask
            
            # Calculate the L2 norm for the pooled vector
            norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-10, norms)
            
            # Apply L2 normalization to the pooled vector
            normalized_batch = mean_pooled / norms
            
            # Append the resulting feature vector to the embeddings list
            embeddings.extend(normalized_batch.tolist())
            
        return embeddings

    def similarity(self, text1: str, text2: str) -> float:
        """Calculates the dot product similarity score between two queries."""
        emb1 = np.array(self.embed_query(text1))
        emb2 = np.array(self.embed_query(text2))
        return float(np.dot(emb1, emb2))