#!/usr/bin/env python3
"""
Unified QwQ GGUF Model
======================
This module provides a single, unified class for loading the QwQ GGUF model
into a PyTorch-compatible architecture using a robust, streaming loader.
"""

import gc
import logging
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

# Import our new sharded loader
from utils.safe_tensor_shard_loader import load_sharded_model

logger = logging.getLogger(__name__)

class QwQModel:
    """
    A unified class to handle the QwQ GGUF model for all tasks.
    This class now correctly uses the StreamingModelLoader to ensure
    memory-safe, direct-to-device loading.
    """
    def __init__(self, model_path: str, device=None):
        """
        Initializes the model lazily. The model is not loaded until it's needed.
        """
        self.model_path = Path(model_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.config = None

    def _lazy_load(self):
        """
        Loads the model using the memory-efficient sharded loader and the tokenizer.
        """
        if self.model is None:
            logger.info(f"Lazy loading model from {self.model_path} using sharded loader...")
            
            # Import the specific model class
            from transformers import AutoModelForCausalLM
            
            # Use the new sharded loader with the correct model class
            self.model = load_sharded_model(str(self.model_path), AutoModelForCausalLM)
            
            # The tokenizer is small and can be loaded normally
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.config = self.model.config
            
            logger.info("Model is fully loaded onto GPU and ready.")

    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        """
        Generates text using the loaded model.
        """
        self._lazy_load() # Ensure model is loaded before use
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model could not be loaded.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def unload(self):
        """
        Explicitly unloads the model and tokenizer to free up GPU memory.
        """
        if self.model is not None:
            logger.info("Unloading model and clearing memory...")
            self.model = None
            self.tokenizer = None
            self.config = None
            
            # Encourage garbage collection
            gc.collect()
            
            # Clear PyTorch's CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Model unloaded and memory cleared.")
        else:
            logger.info("Model is not currently loaded. Nothing to unload.")

    def reload(self):
        """
        Reloads the model. Useful for clearing state or recovering from errors.
        """
        self.unload()
        self._lazy_load()

    def _stitch_attention_windows(self, windows_data: list, window_size: int, stride: int, total_tokens: int) -> dict:
        """
        Stitches overlapping attention windows into complete attention matrices.
        
        Args:
            windows_data: List of window attention data
            window_size: Size of each window
            stride: Stride between windows
            total_tokens: Total number of tokens in the full text
            
        Returns:
            Dictionary with stitched attention matrices for each layer
        """
        if not windows_data:
            return {}
        
        # Get number of layers and heads from first window
        first_window = windows_data[0]
        num_layers = len(first_window['layers'])
        num_heads = first_window['layers'][0]['attention'].shape[0]
        
        # Initialize full attention matrices for each layer
        stitched_layers = []
        
        for layer_idx in range(num_layers):
            # Create full attention matrix filled with zeros
            full_attention = np.zeros((num_heads, total_tokens, total_tokens))
            weight_matrix = np.zeros((total_tokens, total_tokens))
            
            # Process each window
            for window_idx, window_data in enumerate(windows_data):
                start_pos = window_idx * stride
                end_pos = min(start_pos + window_size, total_tokens)
                window_len = end_pos - start_pos
                
                # Get attention for this layer in this window
                window_attention = window_data['layers'][layer_idx]['attention']
                
                # Calculate weights for averaging (higher weight for center of window)
                window_weights = np.ones((window_len, window_len))
                # Apply gaussian-like weighting - center has more weight
                for i in range(window_len):
                    for j in range(window_len):
                        dist_from_center = max(abs(i - window_len/2), abs(j - window_len/2)) / (window_len/2)
                        window_weights[i, j] = 1.0 - 0.5 * dist_from_center
                
                # Add weighted attention to the full matrix
                full_attention[:, start_pos:end_pos, start_pos:end_pos] += window_attention[:, :window_len, :window_len] * window_weights
                weight_matrix[start_pos:end_pos, start_pos:end_pos] += window_weights
            
            # Normalize by weights to get average
            weight_matrix[weight_matrix == 0] = 1  # Avoid division by zero
            full_attention = full_attention / weight_matrix
            
            stitched_layers.append({
                "layer_idx": layer_idx,
                "attention": full_attention,
                "shape": full_attention.shape
            })
        
        return {"layers": stitched_layers}

    def _get_token_positions(self, text: str, tokens: list) -> list:
        """
        Maps tokens back to their character positions in the original text.
        
        Args:
            text: Original text
            tokens: List of tokens
            
        Returns:
            List of (start, end) character positions for each token
        """
        positions = []
        current_pos = 0
        
        for token in tokens:
            # Handle special tokens
            if token.startswith('[') and token.endswith(']'):
                positions.append((current_pos, current_pos))
                continue
                
            # Remove tokenizer artifacts (like ##)
            clean_token = token.replace('##', '').replace('▁', ' ')
            
            # Find token in text
            start = text.find(clean_token, current_pos)
            if start != -1:
                end = start + len(clean_token)
                positions.append((start, end))
                current_pos = end
            else:
                # If not found, use current position
                positions.append((current_pos, current_pos))
        
        return positions

    def extract_attention(self, text: str, window_size: int = 512, use_sliding_window: bool = True, calculator=None) -> list:
        """
        Extracts attention weights using a tiling or sliding window.
        If a calculator is provided, it processes each window's attention data
        as a generator and discards it to save memory.
        
        Args:
            text: Input text to analyze.
            window_size: Size of each attention window.
            use_sliding_window: If False, uses a non-overlapping tiling window.
            calculator: A stateful calculator to process windows one by one.
        
        Returns:
            The result from the calculator if provided, otherwise raw window data.
        """
        self._lazy_load()
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model could not be loaded.")

        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Determine the stride: overlapping (sliding) vs non-overlapping (tiling)
        stride = window_size if not use_sliding_window else window_size // 2
        
        if len(token_ids) <= window_size:
            logger.info(f"Processing text in a single window ({len(token_ids)} tokens)...")
            window_data = self._extract_single_window_attention(text, window_size)
            if calculator and window_data:
                calculator.process_window(window_data)
                return calculator.get_results()
            return [window_data] if window_data else []
        
        logger.info(f"Processing text with {len(token_ids)} tokens using a {'sliding' if use_sliding_window else 'tiling'} window...")
        
        # This part now acts as a generator feeding the calculator
        for window_idx in range(0, len(token_ids), stride):
            start_idx = window_idx
            end_idx = min(start_idx + window_size, len(token_ids))
            window_token_ids = token_ids[start_idx:end_idx]

            if not window_token_ids:
                continue

            window_text = self.tokenizer.decode(window_token_ids, skip_special_tokens=True)
            window_data = self._extract_single_window_attention(window_text, window_size)
            
            if window_data and 'layers' in window_data:
                window_data['metadata'] = {
                    "window_index": window_idx // stride,
                    "token_start_index": start_idx,
                    "token_end_index": end_idx,
                    "text_snippet": window_text[:100] + "..."
                }
                if calculator:
                    calculator.process_window(window_data)
                    del window_data
                    gc.collect()
                else:
                    # If no calculator, this part would collect all data (memory-intensive)
                    # For safety, we'll assume a calculator is intended for long text
                    pass

        if calculator:
            return calculator.get_results()
        return [] # Should not be reached if calculator is used as intended

    def _compute_aggregated_attention(self, layers: list, preserve_heads: bool = True) -> dict:
        """
        Computes aggregated views of attention data.
        
        Args:
            layers: List of layer attention data
            preserve_heads: Whether to keep head dimension
            
        Returns:
            Dictionary with various aggregated views
        """
        # Stack all layer attentions
        all_attentions = np.stack([layer['attention'] for layer in layers])
        # Shape: [num_layers, num_heads, seq_len, seq_len]
        
        aggregated = {
            # Average across all layers and heads
            "mean_attention": np.mean(all_attentions, axis=(0, 1)),
            
            # Average across layers only (keep heads)
            "layer_averaged": np.mean(all_attentions, axis=0) if preserve_heads else None,
            
            # Average across heads only (keep layers)
            "head_averaged": np.mean(all_attentions, axis=1),
            
            # Get attention from specific layers that are often most informative
            "first_layer": all_attentions[0],
            "middle_layer": all_attentions[len(all_attentions)//2],
            "last_layer": all_attentions[-1]
        }
        
        return aggregated

    def _extract_single_window_attention(self, text: str, max_length: int) -> dict:
        """Helper function to extract attention for a single block of text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        
        # Get token strings and positions
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0].cpu().tolist())
        
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids, output_attentions=True)
        
        layers_data = []
        if outputs.attentions:
            for layer_idx, layer_attention in enumerate(outputs.attentions):
                # layer_attention shape: [batch, num_heads, seq_len, seq_len]
                attention_array = layer_attention.cpu().numpy()
                layers_data.append({
                    "layer_idx": layer_idx,
                    "attention": attention_array[0],  # Remove batch dimension, keep [num_heads, seq_len, seq_len]
                    "shape": attention_array[0].shape
                })
        
        return {
            "layers": layers_data,
            "tokens": tokens,
            "sequence_length": len(tokens)
        }

    def get_model_config(self) -> dict:
        """Returns the model's configuration."""
        self._lazy_load() # Ensure model is loaded before use
        return self.config.to_dict()

    def get_embedding(self, text: str) -> np.ndarray:
        """Generates a high-quality embedding for the given text."""
        self._lazy_load() # Ensure model is loaded before use
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model could not be loaded.")
        
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
        self._lazy_load() # Ensure model is loaded before use
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
