#!/usr/bin/env python3
"""
Low-Rank Attention Module
=========================
This module provides efficient attention processing through low-rank decomposition,
semantic steering, and adaptive rank selection.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class LowRankAttention(nn.Module):
    """
    Efficient attention processing with SVD decomposition.
    """
    
    def __init__(self, d_model: int, n_heads: int, max_rank: int = 256):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_rank = max_rank
        self.d_head = d_model // n_heads
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                rank_ratio: float = 0.1) -> torch.Tensor:
        """
        Compute attention with low-rank approximation.
        
        Args:
            query: [batch, seq_len, d_model]
            key: [batch, seq_len, d_model]
            value: [batch, seq_len, d_model]
            rank_ratio: Fraction of ranks to keep
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = query.shape
        
        # Reshape for multi-head attention
        Q = query.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = key.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = value.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_head)
        
        # Apply low-rank decomposition to scores
        scores_lowrank = self._apply_batch_svd(scores, rank_ratio)
        
        # Apply softmax
        attention_weights = torch.softmax(scores_lowrank, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return output
    
    def _apply_batch_svd(self, tensor: torch.Tensor, rank_ratio: float) -> torch.Tensor:
        """
        Apply SVD decomposition to a batch of matrices.
        
        Args:
            tensor: [batch, heads, seq_len, seq_len]
            rank_ratio: Fraction of ranks to keep
            
        Returns:
            Low-rank approximation of input tensor
        """
        batch_size, n_heads, seq_len, _ = tensor.shape
        rank = max(1, int(seq_len * rank_ratio))
        rank = min(rank, self.max_rank, seq_len)
        
        # Process each head separately
        result = torch.zeros_like(tensor)
        
        for b in range(batch_size):
            for h in range(n_heads):
                matrix = tensor[b, h]
                
                # SVD decomposition
                U, S, V = torch.svd(matrix)
                
                # Keep only top-k singular values
                U_r = U[:, :rank]
                S_r = S[:rank]
                V_r = V[:, :rank]
                
                # Reconstruct
                result[b, h] = torch.matmul(U_r, torch.matmul(torch.diag(S_r), V_r.t()))
        
        return result


class SemanticSteering:
    """
    Inject semantic prompts into attention computation for guided processing.
    """
    
    def __init__(self, model, target_layers: List[int] = None):
        """
        Initialize semantic steering.
        
        Args:
            model: The transformer model to steer
            target_layers: List of layer indices to inject prompts into
        """
        self.model = model
        self.target_layers = target_layers or [-4, -3, -2, -1]
        self.steering_cache = {}
        
    def inject_prompts(self, prompts: List[str], tokenizer) -> None:
        """
        Inject steering prompts into model's attention mechanism.
        
        Args:
            prompts: List of steering prompts
            tokenizer: Tokenizer to encode prompts
        """
        # Encode prompts
        prompt_text = " [SEP] ".join(prompts)
        prompt_tokens = tokenizer(prompt_text, return_tensors="pt", 
                                 truncation=True, max_length=128)
        
        # Get prompt embeddings
        with torch.no_grad():
            prompt_embeddings = self.model.get_input_embeddings()(prompt_tokens.input_ids)
        
        # Store in steering cache
        self.steering_cache['prompts'] = prompts
        self.steering_cache['embeddings'] = prompt_embeddings
        self.steering_cache['active'] = True
        
        logger.info(f"Injected {len(prompts)} steering prompts")
        
    def apply_steering(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply semantic steering to hidden states if layer is targeted.
        
        Args:
            layer_idx: Current layer index
            hidden_states: Layer hidden states
            
        Returns:
            Modified hidden states
        """
        if not self.steering_cache.get('active', False):
            return hidden_states
            
        if layer_idx not in self.target_layers:
            return hidden_states
            
        # Concatenate steering embeddings
        steering_embeddings = self.steering_cache['embeddings']
        batch_size = hidden_states.shape[0]
        
        # Expand steering embeddings to match batch size
        steering_embeddings = steering_embeddings.expand(batch_size, -1, -1)
        
        # Concatenate to hidden states
        modified_states = torch.cat([steering_embeddings, hidden_states], dim=1)
        
        return modified_states
    
    def clear_steering(self) -> None:
        """Clear steering cache."""
        self.steering_cache = {}
        logger.info("Cleared semantic steering")


class AdaptiveRankSelector:
    """
    Dynamically select optimal rank based on content complexity.
    """
    
    def __init__(self, min_rank: int = 32, max_rank: int = 512):
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.complexity_history = []
        
    def compute_optimal_rank(self, attention_matrix: np.ndarray, 
                           target_compression: float = 0.9) -> int:
        """
        Compute optimal rank for given attention matrix.
        
        Args:
            attention_matrix: Attention weights
            target_compression: Target compression ratio
            
        Returns:
            Optimal rank
        """
        # Compute SVD
        U, S, Vt = np.linalg.svd(attention_matrix, full_matrices=False)
        
        # Find rank that captures target variance
        total_variance = S.sum()
        cumsum = S.cumsum()
        
        # Find minimum rank that captures (1-target_compression) of variance
        target_variance = total_variance * (1 - target_compression)
        rank = np.argmax(cumsum >= target_variance) + 1
        
        # Apply bounds
        rank = max(self.min_rank, min(rank, self.max_rank))
        
        # Update complexity history
        complexity = rank / attention_matrix.shape[0]
        self.complexity_history.append(complexity)
        
        return rank
    
    def get_average_complexity(self) -> float:
        """Get average complexity from history."""
        if not self.complexity_history:
            return 0.5
        return np.mean(self.complexity_history)


class BlockwiseLowRankAttention(nn.Module):
    """
    Apply different ranks to different blocks for efficiency.
    """
    
    def __init__(self, d_model: int, n_heads: int, block_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.block_size = block_size
        self.base_attention = LowRankAttention(d_model, n_heads)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                adaptive_ranks: bool = True) -> torch.Tensor:
        """
        Process attention in blocks with adaptive rank selection.
        
        Args:
            query, key, value: Input tensors
            adaptive_ranks: Whether to use adaptive rank selection
            
        Returns:
            Output tensor
        """
        batch_size, seq_len, d_model = query.shape
        output = torch.zeros_like(query)
        
        # Process in blocks
        for i in range(0, seq_len, self.block_size):
            for j in range(0, seq_len, self.block_size):
                # Extract blocks
                q_block = query[:, i:i+self.block_size]
                k_block = key[:, j:j+self.block_size]
                v_block = value[:, j:j+self.block_size]
                
                # Determine rank for this block
                if adaptive_ranks:
                    # Higher rank for diagonal blocks, lower for off-diagonal
                    if i == j:
                        rank_ratio = 0.3  # Keep 30% of ranks
                    else:
                        rank_ratio = 0.1  # Keep 10% of ranks
                else:
                    rank_ratio = 0.2  # Fixed ratio
                
                # Compute block attention
                block_output = self.base_attention(q_block, k_block, v_block, rank_ratio)
                
                # Accumulate output
                output[:, i:i+self.block_size] += block_output
        
        return output


class SparseAttentionProcessor:
    """
    Process attention matrices in sparse format for memory efficiency.
    """
    
    def __init__(self, sparsity_threshold: float = 0.01):
        self.sparsity_threshold = sparsity_threshold
        
    def to_sparse(self, attention_matrix: torch.Tensor) -> torch.sparse.FloatTensor:
        """
        Convert attention matrix to sparse format.
        
        Args:
            attention_matrix: Dense attention matrix
            
        Returns:
            Sparse attention matrix
        """
        # Threshold small values
        mask = attention_matrix.abs() > self.sparsity_threshold
        indices = mask.nonzero(as_tuple=False).t()
        values = attention_matrix[mask]
        
        sparse_matrix = torch.sparse.FloatTensor(
            indices, values, attention_matrix.shape
        )
        
        return sparse_matrix
    
    def sparse_softmax(self, sparse_scores: torch.sparse.FloatTensor) -> torch.sparse.FloatTensor:
        """
        Apply softmax to sparse attention scores.
        
        Args:
            sparse_scores: Sparse attention scores
            
        Returns:
            Sparse attention weights
        """
        # Convert to dense for softmax (more efficient sparse softmax is WIP)
        dense_scores = sparse_scores.to_dense()
        dense_weights = torch.softmax(dense_scores, dim=-1)
        
        # Convert back to sparse
        return self.to_sparse(dense_weights)
    
    def get_memory_savings(self, attention_shape: Tuple[int, ...]) -> Dict[str, float]:
        """
        Calculate memory savings from sparse representation.
        
        Args:
            attention_shape: Shape of attention matrix
            
        Returns:
            Dictionary with memory statistics
        """
        total_elements = np.prod(attention_shape)
        
        # Estimate sparsity (typically 90%+ for large sequences)
        estimated_sparsity = 0.9
        sparse_elements = int(total_elements * (1 - estimated_sparsity))
        
        # Memory calculations (float32)
        dense_memory = total_elements * 4  # bytes
        sparse_memory = sparse_elements * 12  # indices + values
        
        return {
            'dense_memory_mb': dense_memory / (1024 * 1024),
            'sparse_memory_mb': sparse_memory / (1024 * 1024),
            'compression_ratio': dense_memory / sparse_memory,
            'sparsity': estimated_sparsity
        }