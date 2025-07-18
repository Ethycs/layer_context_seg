#!/usr/bin/env python3
"""
Graph-Aware Attention Module
============================
This module provides graph-aware attention mechanisms that integrate with
low-rank decomposition, inspired by GAP (Graph-Aware Positional) approach.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

from ..models.low_rank_attention import LowRankAttention

logger = logging.getLogger(__name__)


class GraphAwareAttention(LowRankAttention):
    """
    Graph-aware attention that incorporates adjacency constraints and type embeddings
    while maintaining compatibility with low-rank decomposition.
    """
    
    def __init__(self, d_model: int, n_heads: int, max_rank: int = 256, 
                 n_edge_types: int = 10, type_emb_dim: int = 1):
        """
        Initialize graph-aware attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            max_rank: Maximum rank for low-rank decomposition
            n_edge_types: Number of edge types in the graph
            type_emb_dim: Dimension of type embeddings (typically 1)
        """
        super().__init__(d_model, n_heads, max_rank)
        
        # Type embedding lookup (similar to GAP)
        self.n_edge_types = n_edge_types
        self.type_embedding = nn.Embedding(n_edge_types, type_emb_dim)
        
        # Initialize type embeddings small to not disrupt pre-trained attention
        nn.init.normal_(self.type_embedding.weight, mean=0.0, std=0.01)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                adjacency_matrix: Optional[torch.Tensor] = None,
                edge_types: Optional[torch.Tensor] = None,
                rank_ratio: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute graph-aware attention with optional low-rank approximation.
        
        Args:
            query: [batch, seq_len, d_model]
            key: [batch, seq_len, d_model]
            value: [batch, seq_len, d_model]
            adjacency_matrix: [batch, seq_len, seq_len] binary matrix
            edge_types: [batch, seq_len, seq_len] edge type indices
            rank_ratio: Fraction of ranks to keep for low-rank approximation
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = query.shape
        
        # Reshape for multi-head attention
        Q = query.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = key.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = value.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_head)
        
        # Apply type embeddings if provided (GAP-style)
        if edge_types is not None:
            type_bias = self.type_embedding(edge_types).squeeze(-1)
            # Broadcast type bias to all heads
            scores = scores + type_bias.unsqueeze(1)
        
        # Apply adjacency constraints if provided
        if adjacency_matrix is not None:
            # Create mask from adjacency matrix
            mask = (1.0 - adjacency_matrix) * -10000.0
            scores = scores + mask.unsqueeze(1)
        
        # Apply low-rank decomposition to constrained scores
        scores_lowrank = self._apply_batch_svd(scores, rank_ratio)
        
        # Apply softmax
        attention_weights = torch.softmax(scores_lowrank, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Average attention weights across heads for analysis
        avg_attention = attention_weights.mean(dim=1)
        
        return output, avg_attention


class TypeAwareAttentionProcessor:
    """
    Processes attention with type-aware biasing and graph constraints.
    """
    
    def __init__(self, n_edge_types: int = 10):
        self.n_edge_types = n_edge_types
        self.type_names = [
            "no_relation",
            "explains",
            "elaborates",
            "contradicts",
            "is_example_of",
            "is_consequence_of",
            "depends_on",
            "summarizes",
            "references",
            "continues"
        ]
