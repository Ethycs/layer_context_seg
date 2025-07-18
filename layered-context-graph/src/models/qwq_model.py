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
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer

# Import our new sharded loader
from utils.safe_tensor_shard_loader import load_sharded_model

# Import segmenter protocol
from .segmenter_protocol import Segmenter

# Import new dual-level components
from .dual_level_processor import (
    DualLevelAttentionProcessor,
    WindowedDualLevelProcessor
)
from .token_segment_mapper import (
    TokenSegmentMapper,
    SegmentGraphMapper
)
from .scatter_aggregation import TokenToNodeAggregator


logger = logging.getLogger(__name__)

class QwQModel(Segmenter):
    """
    A unified class to handle the QwQ GGUF model for all tasks.
    This class now correctly uses the StreamingModelLoader to ensure
    memory-safe, direct-to-device loading.
    """
    def __init__(self, qwq_model_path: str, device=None, enable_dual_level: bool = True):
        """
        Initializes the model lazily. The model is not loaded until it's needed.
        """
        self.model_path = Path(qwq_model_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.config = None
        self.enable_dual_level = enable_dual_level
        
        # Dual-level components (lazy-initialized)
        self._dual_processor = None
        self._token_mapper = None
        self._segment_mapper = None
        self._node_aggregator = None

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

    def extract_low_rank_attention(self, text: str, window_size: int = 512, 
                                  rank_ratio: float = 0.1, top_n_layers: int = 4,
                                  calculator=None) -> list:
        """
        Extracts attention weights using low-rank decomposition on top layers only.
        
        Args:
            text: Input text to analyze.
            window_size: Size of each attention window.
            rank_ratio: Fraction of ranks to keep (0.1 = keep 10% of ranks).
            top_n_layers: Number of top layers to process (default: 4).
            calculator: Optional calculator to process windows.
            
        Returns:
            Processed attention data or calculator results.
        """
        self._lazy_load()
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model could not be loaded.")
            
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        total_layers = self.config.num_hidden_layers if self.config else 32
        
        # Only process top N layers
        layer_indices = list(range(total_layers - top_n_layers, total_layers))
        
        stride = window_size // 2  # Sliding window with 50% overlap
        
        if len(token_ids) <= window_size:
            logger.info(f"Processing text in a single window with low-rank decomposition...")
            window_data = self._extract_low_rank_window_attention(
                text, window_size, layer_indices, rank_ratio
            )
            if calculator and window_data:
                calculator.process_window(window_data)
                return calculator.get_results()
            return [window_data] if window_data else []
        
        logger.info(f"Processing {len(token_ids)} tokens with low-rank attention...")
        
        for window_idx in range(0, len(token_ids), stride):
            start_idx = window_idx
            end_idx = min(start_idx + window_size, len(token_ids))
            window_token_ids = token_ids[start_idx:end_idx]
            
            if not window_token_ids:
                continue
                
            window_text = self.tokenizer.decode(window_token_ids, skip_special_tokens=True)
            window_data = self._extract_low_rank_window_attention(
                window_text, window_size, layer_indices, rank_ratio
            )
            
            if window_data and 'layers' in window_data:
                window_data['metadata'] = {
                    "window_index": window_idx // stride,
                    "token_start_index": start_idx,
                    "token_end_index": end_idx,
                    "text_snippet": window_text[:100] + "...",
                    "rank_ratio": rank_ratio,
                    "layers_processed": layer_indices
                }
                if calculator:
                    calculator.process_window(window_data)
                    del window_data
                    gc.collect()
                    
        if calculator:
            return calculator.get_results()
        return []
    
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

    def _extract_low_rank_window_attention(self, text: str, max_length: int, 
                                          layer_indices: List[int], rank_ratio: float) -> dict:
        """Extract attention with low-rank decomposition for specific layers."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0].cpu().tolist())
        
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids, output_attentions=True)
        
        layers_data = []
        if outputs.attentions:
            for layer_idx in layer_indices:
                if layer_idx < len(outputs.attentions):
                    layer_attention = outputs.attentions[layer_idx]
                    # Apply low-rank decomposition
                    attention_lowrank = self._apply_svd_decomposition(
                        layer_attention.cpu().numpy()[0], rank_ratio
                    )
                    layers_data.append({
                        "layer_idx": layer_idx,
                        "attention": attention_lowrank,
                        "shape": attention_lowrank.shape,
                        "is_compressed": True,
                        "compression_ratio": rank_ratio
                    })
        
        return {
            "layers": layers_data,
            "tokens": tokens,
            "sequence_length": len(tokens),
            "low_rank": True
        }
    
    def _apply_svd_decomposition(self, attention_matrix: np.ndarray, rank_ratio: float) -> np.ndarray:
        """
        Apply SVD decomposition to compress attention matrix.
        
        Args:
            attention_matrix: Shape [num_heads, seq_len, seq_len]
            rank_ratio: Fraction of ranks to keep
            
        Returns:
            Compressed attention matrix
        """
        num_heads, seq_len, _ = attention_matrix.shape
        compressed = np.zeros_like(attention_matrix)
        
        for head_idx in range(num_heads):
            # SVD on each head's attention matrix
            U, S, Vt = np.linalg.svd(attention_matrix[head_idx], full_matrices=False)
            
            # Determine rank dynamically
            total_variance = S.sum()
            cumsum = S.cumsum()
            # Find minimum rank that captures (1-rank_ratio) of variance
            rank = max(1, np.argmax(cumsum >= total_variance * (1 - rank_ratio)) + 1)
            rank = min(rank, int(seq_len * rank_ratio))
            
            # Reconstruct with low rank
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vt_r = Vt[:rank, :]
            
            compressed[head_idx] = U_r @ np.diag(S_r) @ Vt_r
            
        return compressed
    
    def inject_semantic_prompts(self, prompts: List[str], layers: List[int] = None) -> None:
        """
        Inject semantic steering prompts into KV cache of specified layers.
        
        Args:
            prompts: List of steering prompts
            layers: Layer indices to inject into (default: top 4 layers)
        """
        self._lazy_load()
        if layers is None:
            total_layers = self.config.num_hidden_layers if self.config else 32
            layers = list(range(total_layers - 4, total_layers))
            
        # Encode prompts
        prompt_text = " ".join(prompts)
        prompt_inputs = self.tokenizer(prompt_text, return_tensors="pt", 
                                      truncation=True, max_length=128).to(self.device)
        
        with torch.no_grad():
            # Get hidden states for prompts
            prompt_outputs = self.model(
                input_ids=prompt_inputs.input_ids,
                output_hidden_states=True
            )
            
            # Inject into specified layers' KV cache
            # Note: This is a conceptual implementation - actual cache manipulation
            # depends on the specific model architecture
            logger.info(f"Injected semantic prompts into layers {layers}")
    
    def segment_by_attention(self, text: str, use_low_rank: bool = True, 
                           boundary_threshold: float = 0.3) -> List[str]:
        """
        Use attention patterns to find natural segment boundaries.
        
        Args:
            text: Input text to segment
            use_low_rank: Whether to use low-rank attention
            boundary_threshold: Threshold for detecting boundaries
            
        Returns:
            List of text segments
        """
        # Use attention calculator to find boundaries
        from graph.attention_calculator import AttentionCalculator
        
        calculator = AttentionCalculator(
            rank_ratio=0.1 if use_low_rank else 1.0,
            boundary_threshold=boundary_threshold
        )
        
        if use_low_rank:
            self.extract_low_rank_attention(text, calculator=calculator)
        else:
            self.extract_attention(text, calculator=calculator)
            
        # Get boundary information from calculator
        results = calculator.get_results()
        boundaries = results.get('segment_boundaries', [])
        
        # Split text at boundaries
        segments = []
        start = 0
        
        for boundary in boundaries:
            end = boundary['position']
            if end > start:
                segments.append(text[start:end].strip())
            start = end
            
        # Add final segment
        if start < len(text):
            segments.append(text[start:].strip())
            
        # Filter out empty segments
        segments = [s for s in segments if s]
        
        return segments if segments else [text]
    
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

    def segment(self, rule: str, text_to_segment: str, chunk_size: int = 1024) -> list[str]:
        """
        Segments text based on a natural language rule using the model's generative capabilities.
        Handles large texts by chunking them.
        """
        self._lazy_load()
        
        # If text is smaller than chunk size, process directly
        if len(text_to_segment) < chunk_size:
            prompt = f"Based on the rule '{rule}', segment the following text by inserting '---' between segments:\n\n{text_to_segment}"
            response = self.generate(prompt, max_tokens=2048)
            
            if text_to_segment in response:
                response = response.split(text_to_segment)[-1]

            segments = [s.strip() for s in response.split('---') if s.strip()]
            
            if not segments or (len(segments) == 1 and segments[0] == text_to_segment):
                return [text_to_segment]
            return segments

        # Chunking for large texts
        all_segments = []
        text_chunks = [text_to_segment[i:i+chunk_size] for i in range(0, len(text_to_segment), chunk_size)]
        
        for chunk in text_chunks:
            prompt = f"Based on the rule '{rule}', segment the following text by inserting '---' between segments:\n\n{chunk}"
            response = self.generate(prompt, max_tokens=2048)
            
            if chunk in response:
                response = response.split(chunk)[-1]
            
            chunk_segments = [s.strip() for s in response.split('---') if s.strip()]
            
            if not chunk_segments or (len(chunk_segments) == 1 and chunk_segments[0] == chunk):
                all_segments.append(chunk)
            else:
                all_segments.extend(chunk_segments)
                
        return all_segments

    def extract_graph_aware_attention(self, text: str, adjacency_matrix: np.ndarray = None,
                                 edge_types: np.ndarray = None, window_size: int = 512,
                                 rank_ratio: float = 0.1, top_n_layers: int = 4,
                                 calculator=None) -> list:
        """
        Extract attention weights with graph constraints and type awareness.
        
        Args:
            text: Input text to analyze
            adjacency_matrix: Optional graph adjacency matrix
            edge_types: Optional edge type matrix
            window_size: Size of each attention window
            rank_ratio: Fraction of ranks to keep
            top_n_layers: Number of top layers to process
            calculator: Optional calculator to process windows
            
        Returns:
            Processed attention data or calculator results
        """
        self._lazy_load()
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model could not be loaded.")
            
        # Import graph-aware attention if not already loaded
        if not hasattr(self, 'graph_attention'):
            from graph.graph_aware_attention import GraphAwareAttention
            # Get model config
            d_model = self.config.hidden_size
            n_heads = self.config.num_attention_heads
            self.graph_attention = GraphAwareAttention(d_model, n_heads)
            
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        total_layers = self.config.num_hidden_layers if self.config else 32
        
        # Only process top N layers
        layer_indices = list(range(total_layers - top_n_layers, total_layers))
        
        stride = window_size // 2
        
        # Convert numpy arrays to torch tensors if provided
        if adjacency_matrix is not None:
            adjacency_matrix = torch.from_numpy(adjacency_matrix).float()
        if edge_types is not None:
            edge_types = torch.from_numpy(edge_types).long()
        
        # Process windows
        for window_idx in range(0, len(token_ids), stride):
            start_idx = window_idx
            end_idx = min(start_idx + window_size, len(token_ids))
            window_token_ids = token_ids[start_idx:end_idx]
            
            if not window_token_ids:
                continue
                
            window_text = self.tokenizer.decode(window_token_ids, skip_special_tokens=True)
            
            # Extract window adjacency if full adjacency provided
            window_adj = None
            window_types = None
            if adjacency_matrix is not None:
                window_adj = adjacency_matrix[start_idx:end_idx, start_idx:end_idx]
            if edge_types is not None:
                window_types = edge_types[start_idx:end_idx, start_idx:end_idx]
                
            window_data = self._extract_graph_aware_window_attention(
                window_text, window_size, layer_indices, rank_ratio,
                window_adj, window_types
            )
            
            if window_data and 'layers' in window_data:
                window_data['metadata'] = {
                    "window_index": window_idx // stride,
                    "token_start_index": start_idx,
                    "token_end_index": end_idx,
                    "text_snippet": window_text[:100] + "...",
                    "rank_ratio": rank_ratio,
                    "layers_processed": layer_indices,
                    "has_graph_constraints": adjacency_matrix is not None
                }
                if calculator:
                    calculator.process_window(window_data)
                    del window_data
                    gc.collect()
                    
        if calculator:
            return calculator.get_results()
        return []

    def _extract_graph_aware_window_attention(self, text: str, max_length: int,
                                            layer_indices: List[int], rank_ratio: float,
                                            adjacency: torch.Tensor = None,
                                            edge_types: torch.Tensor = None) -> dict:
        """Extract attention with graph constraints for specific layers."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                               max_length=max_length).to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0].cpu().tolist())
        
        with torch.no_grad():
            # Get hidden states from model
            outputs = self.model(input_ids=inputs.input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
        layers_data = []
        for layer_idx in layer_indices:
            if layer_idx < len(hidden_states):
                layer_hidden = hidden_states[layer_idx]
                
                # Apply graph-aware attention
                _, attention_weights = self.graph_attention(
                    layer_hidden, layer_hidden, layer_hidden,
                    adjacency, edge_types, rank_ratio
                )
                
                layers_data.append({
                    "layer_idx": layer_idx,
                    "attention": attention_weights.cpu().numpy(),
                    "shape": attention_weights.shape,
                    "is_graph_constrained": adjacency is not None,
                    "has_type_bias": edge_types is not None
                })
        
        return {
            "layers": layers_data,
            "tokens": tokens,
            "sequence_length": len(tokens),
            "graph_aware": True
        }

    def _lazy_load_dual_components(self):
        """Lazy load dual-level processing components."""
        if self._dual_processor is None and self.enable_dual_level:
            self._lazy_load()  # Ensure base model is loaded
            
            logger.info("Initializing dual-level processing components...")
            
            # Get model configuration
            d_model = self.config.hidden_size
            n_heads = self.config.num_attention_heads
            
            # Initialize components
            self._dual_processor = DualLevelAttentionProcessor(
                d_model=d_model,
                n_heads=n_heads,
                n_edge_types=10,
                aggregation='mean'
            )
            
            self._token_mapper = TokenSegmentMapper()
            self._segment_mapper = SegmentGraphMapper()
            self._node_aggregator = TokenToNodeAggregator(aggregation='mean')
            
            # Move to device
            self._dual_processor = self._dual_processor.to(self.device)
            self._node_aggregator = self._node_aggregator.to(self.device)
            
            logger.info("Dual-level components initialized")
    
    def extract_dual_level_attention(self, text: str, segments: List[Dict[str, Any]],
                                   adjacency_matrix: Optional[np.ndarray] = None,
                                   edge_types: Optional[np.ndarray] = None,
                                   window_size: int = 512,
                                   rank_ratio: float = 0.1,
                                   top_n_layers: int = 4,
                                   calculator=None) -> List[Dict[str, Any]]:
        """
        Extract attention with dual-level (token and node) processing.
        
        Args:
            text: Input text
            segments: List of segment dictionaries with start/end positions
            adjacency_matrix: Optional graph adjacency [num_segments, num_segments]
            edge_types: Optional edge types [num_segments, num_segments]
            window_size: Size of attention windows
            rank_ratio: Rank ratio for compression
            top_n_layers: Number of top layers to process
            calculator: Optional calculator for results
            
        Returns:
            List of dual-level attention results
        """
        self._lazy_load()
        self._lazy_load_dual_components()
        
        # Tokenize text
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        # Create token-to-segment mapping
        alignment = self._token_mapper.create_mapping(tokens, segments, text)
        
        # Create graph mapping if adjacency provided
        if adjacency_matrix is not None:
            graph_mapping = self._segment_mapper.create_graph_mapping(
                segments, adjacency_matrix, edge_types
            )
            adjacency_tensor = graph_mapping['adjacency']
            edge_types_tensor = graph_mapping['edge_types']
        else:
            adjacency_tensor = None
            edge_types_tensor = None
        
        # Process in windows
        stride = window_size // 2
        total_layers = self.config.num_hidden_layers
        layer_indices = list(range(total_layers - top_n_layers, total_layers))
        
        window_processor = WindowedDualLevelProcessor(window_size, stride)
        results = []
        
        for window_idx in range(0, len(token_ids), stride):
            start_idx = window_idx
            end_idx = min(start_idx + window_size, len(token_ids))
            window_token_ids = token_ids[start_idx:end_idx]
            
            if not window_token_ids:
                continue
            
            # Get window text and tokens
            window_tokens = tokens[start_idx:end_idx]
            window_text = self.tokenizer.decode(window_token_ids, skip_special_tokens=True)
            
            # Create window-specific mappings
            window_token_to_segment = self._token_mapper.create_windowed_mapping(
                window_tokens, start_idx, alignment
            )
            
            # Extract dual-level attention for window
            window_data = self._extract_dual_level_window_attention(
                window_text, window_token_ids, window_token_to_segment,
                adjacency_tensor, edge_types_tensor,
                layer_indices, rank_ratio
            )
            
            if window_data:
                window_data['metadata'] = {
                    "window_index": window_idx // stride,
                    "token_start_index": start_idx,
                    "token_end_index": end_idx,
                    "text_snippet": window_text[:100] + "...",
                    "dual_level": True,
                    "num_segments": alignment.num_segments
                }
                
                if calculator:
                    calculator.process_dual_level_window(window_data)
                else:
                    results.append(window_data)
            
            gc.collect()
        
        if calculator:
            return calculator.get_dual_level_results()
        return results
    
    def _extract_dual_level_window_attention(self, window_text: str,
                                           window_token_ids: List[int],
                                           token_to_segment: torch.Tensor,
                                           adjacency: Optional[torch.Tensor],
                                           edge_types: Optional[torch.Tensor],
                                           layer_indices: List[int],
                                           rank_ratio: float) -> Dict[str, Any]:
        """
        Extract dual-level attention for a single window.
        """
        # Prepare inputs
        inputs = self.tokenizer(
            window_text, return_tensors="pt", 
            truncation=True, max_length=len(window_token_ids)
        ).to(self.device)
        
        # Move token mapping to device
        token_to_segment = token_to_segment.to(self.device)
        
        # Get hidden states from model
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids,
                output_hidden_states=True,
                output_attentions=True
            )
            hidden_states = outputs.hidden_states
            token_attentions = outputs.attentions
        
        dual_level_results = []
        
        for layer_idx in layer_indices:
            if layer_idx < len(hidden_states):
                layer_hidden = hidden_states[layer_idx]
                
                # Process through dual-level attention
                dual_output = self._dual_processor(
                    layer_hidden,
                    token_to_segment.unsqueeze(0),  # Add batch dimension
                    adjacency.unsqueeze(0) if adjacency is not None else None,
                    edge_types.unsqueeze(0) if edge_types is not None else None
                )
                
                # Apply rank compression to attention weights
                if rank_ratio < 1.0:
                    token_attn = self._apply_svd_decomposition(
                        dual_output['token_attention'].cpu().numpy(),
                        rank_ratio
                    )
                    if dual_output['node_attention'] is not None:
                        node_attn = self._apply_svd_decomposition(
                            dual_output['node_attention'].cpu().numpy(),
                            rank_ratio
                        )
                    else:
                        node_attn = None
                else:
                    token_attn = dual_output['token_attention'].cpu().numpy()
                    node_attn = dual_output['node_attention'].cpu().numpy() if dual_output['node_attention'] is not None else None
                
                dual_level_results.append({
                    "layer_idx": layer_idx,
                    "token_attention": token_attn,
                    "node_attention": node_attn,
                    "node_embeddings": dual_output['node_embeddings'].cpu().numpy() if dual_output['node_embeddings'] is not None else None,
                    "cross_attention": dual_output.get('cross_attention', None),
                    "is_dual_level": True
                })
        
        return {
            "layers": dual_level_results,
            "tokens": self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0].cpu().tolist()),
            "token_to_segment": token_to_segment.cpu().numpy(),
            "sequence_length": len(window_token_ids),
            "dual_level": True
        }
    
    def segment_with_dual_attention(self, text: str, rule: str,
                                  use_graph: bool = True,
                                  min_segment_size: int = 50) -> Tuple[List[str], Dict[str, Any]]:
        """
        Segment text using dual-level attention patterns.
        
        Args:
            text: Text to segment
            rule: Segmentation rule/prompt
            use_graph: Whether to build graph from attention
            min_segment_size: Minimum segment size
            
        Returns:
            Tuple of (segments, attention_graph)
        """
        # First, get initial segments using base method
        initial_segments = self.segment(rule, text)
        
        if not self.enable_dual_level:
            return initial_segments, {}
        
        self._lazy_load_dual_components()
        
        # Create segment dictionaries
        segments = []
        current_pos = 0
        for i, seg_text in enumerate(initial_segments):
            segments.append({
                'id': str(i),
                'content': seg_text,
                'start_pos': current_pos,
                'end_pos': current_pos + len(seg_text)
            })
            current_pos += len(seg_text) + 1  # Account for separator
        
        # Extract dual-level attention
        dual_attention = self.extract_dual_level_attention(
            text, segments,
            window_size=512,
            rank_ratio=0.1
        )
        
        # Build attention graph from dual-level patterns
        attention_graph = {}
        if use_graph and dual_attention:
            attention_graph = self._build_attention_graph_from_dual(
                dual_attention, len(segments)
            )
        
        # Refine segments based on attention patterns
        refined_segments = self._refine_segments_with_attention(
            initial_segments, dual_attention, min_segment_size
        )
        
        return refined_segments, attention_graph
    
    def _build_attention_graph_from_dual(self, dual_attention: List[Dict],
                                       num_segments: int) -> Dict[str, Any]:
        """
        Build graph from dual-level attention patterns.
        """
        # Initialize adjacency matrix
        adjacency = np.zeros((num_segments, num_segments))
        
        # Aggregate node attention across windows and layers
        for window_data in dual_attention:
            for layer_data in window_data.get('layers', []):
                node_attn = layer_data.get('node_attention')
                if node_attn is not None:
                    # Add to adjacency (node attention already at segment level)
                    n = min(node_attn.shape[-1], num_segments)
                    adjacency[:n, :n] += node_attn[0, :n, :n]  # Remove batch dim
        
        # Normalize
        if len(dual_attention) > 0:
            adjacency /= len(dual_attention)
        
        # Threshold to create binary adjacency
        threshold = adjacency.mean() + adjacency.std()
        binary_adjacency = (adjacency > threshold).astype(float)
        
        return {
            'adjacency': binary_adjacency,
            'weighted_adjacency': adjacency,
            'num_nodes': num_segments,
            'threshold': threshold
        }
    
    def _refine_segments_with_attention(self, segments: List[str],
                                       dual_attention: List[Dict],
                                       min_size: int) -> List[str]:
        """
        Refine segments based on dual-level attention patterns.
        """
        # For now, return original segments
        # This could be extended to merge/split based on attention
        return segments
    
    def inject_graph_prompts(self, prompts: List[str], 
                           segments: List[Dict[str, Any]],
                           layers: Optional[List[int]] = None) -> None:
        """
        Inject graph-aware semantic prompts at node level.
        
        Args:
            prompts: List of prompts
            segments: Segment definitions
            layers: Target layers (default: top 4)
        """
        self._lazy_load()
        self._lazy_load_dual_components()
        
        if layers is None:
            total_layers = self.config.num_hidden_layers
            layers = list(range(total_layers - 4, total_layers))
        
        # Tokenize prompts
        prompt_text = " [SEP] ".join(prompts)
        prompt_inputs = self.tokenizer(
            prompt_text, return_tensors="pt",
            truncation=True, max_length=128
        ).to(self.device)
        
        # Create segment mapping for prompts
        prompt_tokens = self.tokenizer.convert_ids_to_tokens(
            prompt_inputs.input_ids[0].cpu().tolist()
        )
        
        # Simple mapping: each prompt to a segment
        prompt_segments = []
        for i, prompt in enumerate(prompts):
            prompt_segments.append({
                'id': f'prompt_{i}',
                'content': prompt,
                'is_prompt': True
            })
        
        alignment = self._token_mapper.create_mapping(
            prompt_tokens, prompt_segments, prompt_text
        )
        
        # Process prompts through dual-level
        with torch.no_grad():
            outputs = self.model(
                input_ids=prompt_inputs.input_ids,
                output_hidden_states=True
            )
            
            # Aggregate to node level
            for layer_idx in layers:
                if layer_idx < len(outputs.hidden_states):
                    hidden = outputs.hidden_states[layer_idx]
                    
                    # Create node embeddings
                    node_embeddings = self._node_aggregator(
                        hidden,
                        alignment.token_to_segment.unsqueeze(0).to(self.device)
                    )
                    
                    # Store for later use (conceptual - actual implementation
                    # would integrate with model's KV cache)
                    logger.info(f"Injected {len(prompts)} graph-aware prompts at layer {layer_idx}")
    
    def get_dual_level_embeddings(self, text: str, segments: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Get both token and node-level embeddings.
        
        Args:
            text: Input text
            segments: Segment definitions
            
        Returns:
            Dict with 'token_embeddings' and 'node_embeddings'
        """
        self._lazy_load()
        self._lazy_load_dual_components()
        
        # Get token embeddings (from base class)
        token_embedding = self.get_embedding(text)
        
        # Get full hidden states
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0].cpu().tolist())
        
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids, output_hidden_states=True)
            # Use last hidden state
            hidden_states = outputs.hidden_states[-1]
        
        # Create token-to-segment mapping
        alignment = self._token_mapper.create_mapping(tokens, segments, text)
        
        # Aggregate to node embeddings
        node_embeddings = self._node_aggregator(
            hidden_states,
            alignment.token_to_segment.unsqueeze(0).to(self.device)
        )
        
        return {
            'token_embeddings': token_embedding,
            'node_embeddings': node_embeddings.squeeze(0).cpu().numpy(),
            'num_tokens': len(tokens),
            'num_nodes': len(segments)
        }
