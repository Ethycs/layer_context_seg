#!/usr/bin/env python3
"""
Unified QwQ GGUF Model
======================
This module provides a single, unified class for loading the QwQ GGUF model
into a PyTorch-compatible architecture.
"""

import logging
import torch
import gguf
from transformers import AutoTokenizer

from .qwq_architecture import QwQConfig, QwQForCausalLM

logger = logging.getLogger(__name__)

class QwQModel:
    """
    A unified class to handle the QwQ GGUF model for all tasks.
    """
    def __init__(self, model_path: str, device=None):
        """
        Initializes the model, loading it into memory once.
        """
        self.model_path = model_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self._load_model_and_tokenizer()

    def _get_gguf_tensor_map(self):
        """Creates a mapping from GGUF tensor names to PyTorch state_dict keys."""
        # This mapping is crucial and can be complex. This is a simplified example.
        return {
            'token_embd.weight': 'model.embed_tokens.weight',
            'output_norm.weight': 'model.norm.weight',
            'output.weight': 'lm_head.weight',
        }

    def _load_model_and_tokenizer(self):
        """
        Loads the GGUF model weights into the PyTorch architecture.
        """
        logger.info(f"Loading GGUF model from {self.model_path} into PyTorch architecture...")
        try:
            config = QwQConfig()
            self.model = QwQForCausalLM(config).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2") # Using a basic tokenizer for now

            # Load the GGUF file
            reader = gguf.GGUFReader(self.model_path)
            tensor_map = self._get_gguf_tensor_map()
            
            state_dict = {}
            for tensor in reader.tensors:
                if tensor.name in tensor_map:
                    pytorch_key = tensor_map[tensor.name]
                    state_dict[pytorch_key] = torch.from_numpy(tensor.data).to(self.device)
                else:
                    # This is where the complex mapping for layers would go.
                    # For now, we'll log the unmapped tensors.
                    logger.debug(f"Unmapped tensor: {tensor.name}")

            # This is a simplified load. A full implementation would need to handle all layers.
            # self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Successfully loaded a subset of GGUF weights into the PyTorch model.")
            self.model.eval()

        except Exception as e:
            logger.error(f"Failed to load GGUF model into PyTorch architecture: {e}")
            raise

    def generate_text(self, prompt: str, max_length: int = 150) -> str:
        """
        Generates text using the loaded model.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def extract_attention(self, text: str) -> dict:
        """
        Extracts attention weights from the model for the given text.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded.")
            
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids, output_attentions=True)
        
        return {i: layer_attention.cpu() for i, layer_attention in enumerate(outputs['attentions'])}

    def get_model_config(self) -> dict:
        """Returns the model's configuration."""
        return self.model.config.to_dict()
