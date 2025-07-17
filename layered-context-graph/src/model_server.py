#!/usr/bin/env python3
"""
QwQ Model Server (FastAPI)
==========================
This script loads the QwQModel into VRAM and serves it via a high-performance
FastAPI server. This allows the main application to crash and restart
without reloading the large model.
"""

import logging
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from models.qwq_model import QwQModel

# --- Configuration -------------------------------------------------
HOST = "127.0.0.1"
PORT = 5001
MODEL_PATH = "/workspaces/layer_context_seg/qwq.gguf" # This should be configured more robustly
# -------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Data Models ---------------------------------------------------
class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 150

class AttentionRequest(BaseModel):
    text: str

# --- Application Lifecycle -----------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup and clear it on shutdown."""
    logger.info("--- Model Server Lifespan: Startup ---")
    logger.info("Loading model for the first time...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    app.state.model = QwQModel(MODEL_PATH, device)
    logger.info("Model is fully loaded and ready to serve.")
    yield
    logger.info("--- Model Server Lifespan: Shutdown ---")
    app.state.model = None
    # Add any other cleanup here, e.g., torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

# --- API Endpoints -------------------------------------------------
@app.post("/generate")
async def generate(request: GenerationRequest):
    """Handle text generation requests."""
    if not app.state.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        generated_text = app.state.model.generate(request.prompt, max_tokens=request.max_tokens)
        return {"generated_text": generated_text}
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_attention")
async def extract_attention(request: AttentionRequest):
    """Handle attention extraction requests."""
    if not app.state.model:
        logger.error("Model not loaded, cannot process request.")
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        logger.info(f"Received attention extraction request for text: '{request.text[:80]}...'")
        
        # The model now returns a JSON-serializable dictionary
        attention_data = app.state.model.extract_attention(request.text)
        
        # Log the structure of the returned data for debugging
        attentions = attention_data.get("attentions", [])
        num_layers = len(attentions)
        logger.info(f"Extracted {num_layers} layers of attention.")
        
        if num_layers > 0:
            # Use numpy to safely inspect shape and mean
            first_layer = np.array(attentions[0])
            logger.info(f"First layer shape: {first_layer.shape}")
            logger.info(f"First layer sample (first 5 elements of first row): {first_layer[0, :5]}")
        
        # Create a summary from the returned data
        layer_0_shape = np.array(attentions[0]).shape if num_layers > 0 else None
        layer_0_mean = np.mean(attentions[0]) if num_layers > 0 else None

        attention_summary = {
            "num_layers": num_layers,
            "layer_0_shape": list(layer_0_shape) if layer_0_shape else None,
            "layer_0_mean": float(layer_0_mean) if layer_0_mean is not None else None,
        }
        
        logger.info("Successfully created attention summary.")
        
        # Return the full data and the summary
        return {
            "status": "success", 
            "attention_summary": attention_summary,
            "attentions": attentions
        }
    except Exception as e:
        logger.error(f"Error during attention extraction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
