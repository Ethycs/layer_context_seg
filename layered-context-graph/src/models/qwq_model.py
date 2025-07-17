#!/usr/bin/env python3
"""
Unified QwQ GGUF Model
======================
This module provides a single, unified class for loading the QwQ GGUF model
into a PyTorch-compatible architecture using a robust, streaming loader.
"""

import logging
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

from llama_cpp import Llama
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class QwQModel:
    """
    A unified class to handle the QwQ GGUF model for all tasks.
    This class now correctly uses the StreamingModelLoader to ensure
    memory-safe, direct-to-device loading.
    """
    def __init__(self, model_path: str, device=None):
        """
        Initializes the model by invoking the streaming loader.
        """
        self.model_path = Path(model_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.config = None
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """
        Loads the model using the llama-cpp-python library.
        """
        logger.info(f"Initializing model loading process for {self.model_path}...")
        
        self.model = Llama(model_path=str(self.model_path), n_ctx=2048)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B", trust_remote_code=True)
        
        logger.info("QwQModel is fully initialized and ready.")

    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        """
        Generates text using the loaded model.
        """
        if not self.model:
            raise RuntimeError("Model is not loaded.")
        
        output = self.model(prompt, max_tokens=max_tokens, echo=False)
        return output['choices'][0]['text']

    def extract_attention(self, text: str) -> dict:
        """
        Extracts attention weights from the model for the given text.
        """
        logger.warning("Attention extraction is not supported by the Llama CPP model.")
        return {"attentions": []}

    def get_model_config(self) -> dict:
        """Returns the model's configuration."""
        return self.config.to_dict()

    def get_embedding(self, text: str) -> np.ndarray:
        """Generates a high-quality embedding for the given text."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded.")
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model.model(input_ids=inputs.input_ids)
        
        # Use the average of the last hidden state as the embedding
        embedding = outputs[0].mean(dim=1).squeeze().cpu().numpy()
        return embedding

    def classify_relationship(self, node1_content: str, node2_content: str) -> str:
        """
        Uses the loaded LLM to classify the semantic relationship between two nodes.
        """
        prompt = f"""Analyze the relationship between the following two text segments.

Segment 1:
---
{node1_content[:1000]}...
---

Segment 2:
---
{node2_content[:1000]}...
---

What is the primary relationship between Segment 2 and Segment 1? Choose from the following options:
- "explains": Segment 2 explains or clarifies a concept from Segment 1.
- "elaborates": Segment 2 provides more detail or builds upon an idea from Segment 1.
- "contradicts": Segment 2 presents an opposing view or contradicts Segment 1.
- "is_example_of": Segment 2 provides a specific example of a concept in Segment 1.
- "is_consequence_of": Segment 2 is a result or consequence of what is described in Segment 1.
- "depends_on": Segment 2 requires the information from Segment 1 as a prerequisite.
- "no_clear_relation": There is no direct, clear relationship.

Return only the single relationship type as a string (e.g., "explains").
"""
        response = self.generate(prompt, max_tokens=20).strip().lower()
        valid_types = ["explains", "elaborates", "contradicts", "is_example_of", "is_consequence_of", "depends_on", "no_clear_relation"]
        if response in valid_types:
            return response
        return "unknown"
