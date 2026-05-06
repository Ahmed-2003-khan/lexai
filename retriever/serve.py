import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from retriever.engine import DPRInferenceEngine

app = FastAPI(title="LexAI Retriever Microservice")

# Global engine instance
engine = None

class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedding: List[float]

@app.on_event("startup")
async def startup_event():
    global engine
    try:
        query_model = os.getenv("QUERY_ENCODER_PATH", "models/dpr/query_encoder.onnx")
        passage_model = os.getenv("PASSAGE_ENCODER_PATH", "models/dpr/passage_encoder.onnx")
        tokenizer_path = os.getenv("TOKENIZER_PATH", "models/dpr/tokenizer")
        
        # Check if models exist before loading
        if os.path.exists(query_model) and os.path.exists(tokenizer_path):
            engine = DPRInferenceEngine(query_model, passage_model, tokenizer_path)
            print("✅ DPR Engine loaded successfully in retriever service.")
        else:
            print("⚠️ Warning: Model files not found. Waiting for ingestion step.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

@app.get("/health")
def health_check():
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    return {"status": "ok", "service": "retriever"}

@app.post("/embed", response_model=EmbedResponse)
def embed_query(request: EmbedRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized. Ensure models exist.")
    
    # Generate embedding using the local ONNX engine
    embedding = engine.embed_query(request.text)
    
    # Ensure it's a standard python list for JSON serialization
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
        
    return {"embedding": embedding}

if __name__ == "__main__":
    # Run the microservice on port 8001 as defined in docker-compose
    uvicorn.run(app, host="0.0.0.0", port=8001)