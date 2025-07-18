#!/usr/bin/env python3
"""
Dual-Level Processor
====================
Implements dual-level (token and node) processing for GAP-style architectures.
Processes attention at both token and node levels with cross-level information exchange.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

# Import scatter operations
from .scatter_aggregation import (
    TokenToNodeAggregator, 
    NodeToTokenDistributor,
    scatter_mean
)

logger = logging.getLogger(__name__)


class DualLevelAttentionProcessor(nn.Module):
    """
    Processes attention at both token and node levels with cross-level interaction.
    Core component for GAP-style graph-aware transformers.
    """
    
    def __init__(self, d_model: int, n_heads: int, n_edge_types: int = 10,
                 aggregation: str = 'mean', combine_method: str = 'add'):
        """
        Initialize dual-level processor.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_edge_types: Number of edge types for graph
            aggregation: Method for token-to-node aggregation
            combine_method: Method for combining token and node features
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Token-level attention (standard)
        self.token_attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        
        # Node-level attention (graph-aware)
        from layered_context_graph.src.models.graph_aware_attention import GraphAwareAttention
        self.node_attention = GraphAwareAttention(
            d_model, n_heads, n_edge_types=n_edge_types
        )
        
        # Aggregation and distribution
        self.token_to_node = TokenToNodeAggregator(aggregation)
        self.node_to_token = NodeToTokenDistributor(combine=combine_method)
        
        # Cross-level interaction
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        
        # Layer norms
        self.token_norm = nn.LayerNorm(d_model)
        self.node_norm = nn.LayerNorm(d_model)
        
        # Projection for final output
        self.output_projection = nn.Linear(d_model, d_model)
    
    def forward(self, hidden_states: torch.Tensor,
                token_to_segment: torch.Tensor,
                adjacency_matrix: Optional[torch.Tensor] = None,
                edge_types: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process attention at both token and node levels.
        
        Args:
            hidden_states: Token representations [batch, seq_len, d_model]
            token_to_segment: Mapping from tokens to segments/nodes [batch, seq_len]
            adjacency_matrix: Graph adjacency for nodes [batch, num_nodes, num_nodes]
            edge_types: Edge type matrix [batch, num_nodes, num_nodes]
            attention_mask: Token-level attention mask [batch, seq_len]
            
        Returns:
            Dict containing:
                - 'token_output': Enhanced token representations
                - 'node_embeddings': Node-level representations
                - 'token_attention': Token-level attention weights
                - 'node_attention': Node-level attention weights
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Step 1: Token-level attention
        token_output, token_attention_weights = self.token_attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=attention_mask
        )
        token_output = self.token_norm(hidden_states + token_output)
        
        # Step 2: Aggregate tokens to nodes
        node_embeddings = self.token_to_node(
            token_output, token_to_segment,
            attention_weights=token_attention_weights.mean(dim=1)  # Average across heads
        )
        
        # Step 3: Node-level attention (with graph constraints)
        if adjacency_matrix is not None:
            node_output, node_attention_weights = self.node_attention(
                node_embeddings, node_embeddings, node_embeddings,
                adjacency_matrix=adjacency_matrix,
                edge_types=edge_types
            )
            node_embeddings = self.node_norm(node_embeddings + node_output)
        else:
            # Fall back to standard attention if no graph
            node_output, node_attention_weights = self.token_attention(
                node_embeddings, node_embeddings, node_embeddings
            )
            node_embeddings = self.node_norm(node_embeddings + node_output)
        
        # Step 4: Cross-level attention (tokens attend to nodes)
        cross_output, cross_weights = self.cross_attention(
            token_output, node_embeddings, node_embeddings
        )
        token_output = token_output + cross_output
        
        # Step 5: Distribute node information back to tokens
        enhanced_tokens = self.node_to_token(
            node_embeddings, token_output, token_to_segment
        )
        
        # Final projection
        output = self.output_projection(enhanced_tokens)
        
        return {
            'token_output': output,
            'node_embeddings': node_embeddings,
            'token_attention': token_attention_weights,
            'node_attention': node_attention_weights,
            'cross_attention': cross_weights
        }


class DualLevelEncoder(nn.Module):
    """
    Full encoder with multiple dual-level layers.
    Replaces standard transformer encoder for graph-aware processing.
    """
    
    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 n_edge_types: int = 10, dropout: float = 0.1):
        """
        Initialize dual-level encoder.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            n_edge_types: Number of edge types
            dropout: Dropout rate
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            DualLevelEncoderLayer(d_model, n_heads, n_edge_types, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, hidden_states: torch.Tensor,
                token_to_segment: torch.Tensor,
                adjacency_matrix: Optional[torch.Tensor] = None,
                edge_types: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Process through all encoder layers.
        
        Returns:
            Dict with final outputs and per-layer attention weights
        """
        all_token_attentions = []
        all_node_attentions = []
        all_node_embeddings = []
        
        for layer in self.layers:
            layer_output = layer(
                hidden_states, token_to_segment,
                adjacency_matrix, edge_types, attention_mask
            )
            
            hidden_states = layer_output['token_output']
            all_token_attentions.append(layer_output['token_attention'])
            all_node_attentions.append(layer_output['node_attention'])
            all_node_embeddings.append(layer_output['node_embeddings'])
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        return {
            'last_hidden_state': hidden_states,
            'node_embeddings': all_node_embeddings[-1],
            'all_token_attentions': all_token_attentions,
            'all_node_attentions': all_node_attentions,
            'all_node_embeddings': all_node_embeddings
        }


class DualLevelEncoderLayer(nn.Module):
    """
    Single encoder layer with dual-level processing.
    """
    
    def __init__(self, d_model: int, n_heads: int, n_edge_types: int = 10,
                 dropout: float = 0.1, d_ff: int = None):
        """
        Initialize encoder layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_edge_types: Number of edge types
            dropout: Dropout rate
            d_ff: Feed-forward dimension (defaults to 4*d_model)
        """
        super().__init__()
        
        d_ff = d_ff or 4 * d_model
        
        # Dual-level attention
        self.dual_attention = DualLevelAttentionProcessor(
            d_model, n_heads, n_edge_types
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states: torch.Tensor,
                token_to_segment: torch.Tensor,
                adjacency_matrix: Optional[torch.Tensor] = None,
                edge_types: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process through one encoder layer.
        """
        # Dual-level attention with residual
        attn_output = self.dual_attention(
            hidden_states, token_to_segment,
            adjacency_matrix, edge_types, attention_mask
        )
        
        hidden_states = self.norm1(hidden_states + self.dropout(attn_output['token_output']))
        
        # Feed-forward with residual
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.norm2(hidden_states + ffn_output)
        
        return {
            'token_output': hidden_states,
            'node_embeddings': attn_output['node_embeddings'],
            'token_attention': attn_output['token_attention'],
            'node_attention': attn_output['node_attention']
        }


class WindowedDualLevelProcessor:
    """
    Handles dual-level processing for windowed attention.
    Manages token-segment mapping across sliding windows.
    """
    
    def __init__(self, window_size: int = 512, stride: int = 256):
        """
        Initialize windowed processor.
        
        Args:
            window_size: Size of attention window
            stride: Stride for sliding windows
        """
        self.window_size = window_size
        self.stride = stride
        self.segment_cache = {}
    
    def process_window(self, window_tokens: List[str], 
                      window_start: int,
                      global_segments: Dict[str, Any]) -> torch.Tensor:
        """
        Create token-to-segment mapping for a window.
        
        Args:
            window_tokens: Tokens in current window
            window_start: Start position in global sequence
            global_segments: Global segment information
            
        Returns:
            Token-to-segment mapping for window
        """
        window_size = len(window_tokens)
        token_to_segment = torch.zeros(window_size, dtype=torch.long)
        
        # Find which segments overlap with this window
        window_end = window_start + window_size
        
        for seg_id, segment in global_segments.items():
            seg_start = segment.get('token_start', 0)
            seg_end = segment.get('token_end', 0)
            
            # Check overlap
            overlap_start = max(seg_start, window_start)
            overlap_end = min(seg_end, window_end)
            
            if overlap_start < overlap_end:
                # Map tokens in overlap to this segment
                local_start = overlap_start - window_start
                local_end = overlap_end - window_start
                token_to_segment[local_start:local_end] = int(seg_id)
        
        return token_to_segment
    
    def merge_node_embeddings(self, window_embeddings: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Merge node embeddings from overlapping windows.
        
        Args:
            window_embeddings: List of node embeddings from each window
            
        Returns:
            Merged node embeddings
        """
        # Collect all unique nodes
        all_nodes = set()
        for window in window_embeddings:
            all_nodes.update(window['node_ids'])
        
        num_nodes = len(all_nodes)
        hidden_dim = window_embeddings[0]['embeddings'].shape[-1]
        
        # Initialize merged embeddings
        merged = torch.zeros(num_nodes, hidden_dim)
        counts = torch.zeros(num_nodes)
        
        # Accumulate embeddings
        for window in window_embeddings:
            node_ids = window['node_ids']
            embeddings = window['embeddings']
            
            for i, node_id in enumerate(node_ids):
                merged[node_id] += embeddings[i]
                counts[node_id] += 1
        
        # Average
        counts = counts.clamp(min=1)
        merged = merged / counts.unsqueeze(-1)
        
        return merged


class AdaptiveDualLevelProcessor(nn.Module):
    """
    Adaptive processor that switches between token-only and dual-level
    based on sequence length and graph density.
    """
    
    def __init__(self, d_model: int, n_heads: int, 
                 length_threshold: int = 1024,
                 density_threshold: float = 0.3):
        """
        Initialize adaptive processor.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            length_threshold: Sequence length to trigger dual-level
            density_threshold: Graph density to trigger node-level
        """
        super().__init__()
        
        self.length_threshold = length_threshold
        self.density_threshold = density_threshold
        
        # Token-only path
        self.token_only = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        
        # Dual-level path
        self.dual_level = DualLevelAttentionProcessor(
            d_model, n_heads
        )
    
    def forward(self, hidden_states: torch.Tensor,
                token_to_segment: Optional[torch.Tensor] = None,
                adjacency_matrix: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Adaptively process based on input characteristics.
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Check if we should use dual-level
        use_dual_level = False
        
        if seq_len > self.length_threshold:
            use_dual_level = True
        
        if adjacency_matrix is not None:
            density = adjacency_matrix.float().mean()
            if density > self.density_threshold:
                use_dual_level = True
        
        if use_dual_level and token_to_segment is not None:
            # Use dual-level processing
            return self.dual_level(
                hidden_states, token_to_segment,
                adjacency_matrix, **kwargs
            )
        else:
            # Use token-only processing
            output, weights = self.token_only(
                hidden_states, hidden_states, hidden_states
            )
            return {
                'token_output': output,
                'node_embeddings': None,
                'token_attention': weights,
                'node_attention': None
            }


def create_hierarchical_dual_processor(num_levels: int = 3) -> nn.ModuleList:
    """
    Create processors for hierarchical dual-level processing.
    
    Args:
        num_levels: Number of hierarchy levels
        
    Returns:
        List of processors for each level
    """
    processors = nn.ModuleList()
    
    for level in range(num_levels):
        # Higher levels get more heads and parameters
        n_heads = 8 * (2 ** level)
        d_model = 768  # Base dimension
        
        processor = DualLevelAttentionProcessor(
            d_model=d_model,
            n_heads=min(n_heads, 32),  # Cap at 32 heads
            aggregation='mean' if level == 0 else 'attention'
        )
        processors.append(processor)
    
    return processors