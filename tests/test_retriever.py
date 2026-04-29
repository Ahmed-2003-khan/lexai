import os
import pytest
import numpy as np

try:
    from retriever.engine import DPRInferenceEngine
except ImportError:
    DPRInferenceEngine = None

MODEL_DIR = "models/dpr"
HAS_MODELS = os.path.exists(os.path.join(MODEL_DIR, "query_encoder.onnx"))

@pytest.fixture
def engine():
    if not HAS_MODELS:
        pytest.skip("ONNX models not found in models/dpr/. Skipping tests.")
    return DPRInferenceEngine(
        query_onnx_path=os.path.join(MODEL_DIR, "query_encoder.onnx"),
        passage_onnx_path=os.path.join(MODEL_DIR, "passage_encoder.onnx"),
        tokenizer_path=os.path.join(MODEL_DIR, "tokenizer")
    )

@pytest.mark.skipif(not HAS_MODELS, reason="ONNX models not found")
def test_engine_loads(engine):
    assert engine.query_session is not None
    assert engine.passage_session is not None
    assert engine.tokenizer is not None

@pytest.mark.skipif(not HAS_MODELS, reason="ONNX models not found")
def test_embed_query_dim(engine):
    vec = engine.embed_query("test query string")
    assert isinstance(vec, list)
    assert len(vec) == 768

@pytest.mark.skipif(not HAS_MODELS, reason="ONNX models not found")
def test_embed_is_normalized(engine):
    vec = engine.embed_passage("test passage string")
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 0.001

@pytest.mark.skipif(not HAS_MODELS, reason="ONNX models not found")
def test_similarity_positive_pair(engine):
    text1 = "How to write a valid contract?"
    text2 = "Requirements for forming a legally binding agreement."
    
    score = engine.similarity(text1, text2)
    assert score > 0.5

@pytest.mark.skipif(not HAS_MODELS, reason="ONNX models not found")
def test_similarity_negative_pair(engine):
    base_text = "How to write a valid contract?"
    similar_text = "Requirements for forming a legally binding agreement."
    unrelated_text = "The recipe for making chocolate chip cookies."
    
    positive_score = engine.similarity(base_text, similar_text)
    negative_score = engine.similarity(base_text, unrelated_text)
    
    assert negative_score < positive_score