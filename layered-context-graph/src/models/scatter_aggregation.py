#!/usr/bin/env python3
"""
Scatter Aggregation Module
==========================
Implements scatter operations for token-to-node aggregation,
essential for GAP-style dual-level processing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, 
                 dim_size: Optional[int] = None) -> torch.Tensor:
    """
    Scatter mean operation - averages values by index.
    
    Args:
        src: Source tensor to aggregate [*dims]
        index: Index tensor mapping src elements to groups [*dims[:dim], src.shape[dim]]
        dim: Dimension to scatter along
        dim_size: Size of output dimension (defaults to max(index)+1)
        
    Returns:
        Aggregated tensor with mean values per group
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1
    
    # Create output tensor
    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    
    # Count elements per group for averaging
    ones = torch.ones_like(src)
    count = torch.zeros(shape, dtype=src.dtype, device=src.device)
    
    # Scatter add
    out.scatter_add_(dim, index, src)
    count.scatter_add_(dim, index, ones)
    
    # Avoid division by zero
    count = count.clamp(min=1)
    
    # Compute mean
    return out / count


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                dim_size: Optional[int] = None) -> torch.Tensor:
    """
    Scatter sum operation - sums values by index.
    
    Args:
        src: Source tensor to aggregate
        index: Index tensor mapping src elements to groups
        dim: Dimension to scatter along
        dim_size: Size of output dimension
        
    Returns:
        Aggregated tensor with sum per group
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1
    
    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    
    return out.scatter_add_(dim, index, src)


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scatter max operation - finds maximum values by index.
    
    Args:
        src: Source tensor to aggregate
        index: Index tensor mapping src elements to groups
        dim: Dimension to scatter along
        dim_size: Size of output dimension
        
    Returns:
        Tuple of (max_values, argmax_indices)
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1
    
    shape = list(src.shape)
    shape[dim] = dim_size
    
    # Initialize with very negative values
    out = torch.full(shape, float('-inf'), dtype=src.dtype, device=src.device)
    argmax = torch.zeros(shape, dtype=torch.long, device=src.device)
    
    # Scatter max operation
    for i in range(src.shape[dim]):
        mask = index == i
        if mask.any():
            values = src.masked_select(mask)
            max_val, max_idx = values.max()
            out[..., i] = max_val
            argmax[..., i] = max_idx
    
    return out, argmax


class TokenToNodeAggregator(nn.Module):
    """
    Aggregates token-level representations to node-level representations.
    Supports multiple aggregation strategies.
    """
    
    def __init__(self, aggregation: str = 'mean', learnable_weights: bool = False):
        """
        Initialize aggregator.
        
        Args:
            aggregation: Aggregation method ('mean', 'sum', 'max', 'attention')
            learnable_weights: Whether to use learnable aggregation weights
        """
        super().__init__()
        self.aggregation = aggregation
        self.learnable_weights = learnable_weights
        
        if learnable_weights:
            # Learnable temperature for attention-based aggregation
            self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, token_embeddings: torch.Tensor, 
                token_to_node: torch.Tensor,
                attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aggregate token embeddings to node embeddings.
        
        Args:
            token_embeddings: Token representations [batch, seq_len, hidden_dim]
            token_to_node: Mapping from tokens to nodes [batch, seq_len]
            attention_weights: Optional attention weights for weighted aggregation
            
        Returns:
            Node embeddings [batch, num_nodes, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = token_embeddings.shape
        num_nodes = int(token_to_node.max()) + 1
        
        # Reshape for scatter operations
        token_embeddings_flat = token_embeddings.view(-1, hidden_dim)
        token_to_node_flat = token_to_node.view(-1)
        
        if self.aggregation == 'mean':
            # Simple mean aggregation
            node_embeddings_flat = scatter_mean(
                token_embeddings_flat, token_to_node_flat, 
                dim=0, dim_size=num_nodes
            )
        
        elif self.aggregation == 'sum':
            # Sum aggregation
            node_embeddings_flat = scatter_sum(
                token_embeddings_flat, token_to_node_flat,
                dim=0, dim_size=num_nodes
            )
        
        elif self.aggregation == 'max':
            # Max pooling aggregation
            node_embeddings_flat, _ = scatter_max(
                token_embeddings_flat, token_to_node_flat,
                dim=0, dim_size=num_nodes
            )
        
        elif self.aggregation == 'attention' and attention_weights is not None:
            # Attention-weighted aggregation
            attention_flat = attention_weights.view(-1, 1)
            weighted_embeddings = token_embeddings_flat * attention_flat
            
            # Sum weighted embeddings
            weighted_sum = scatter_sum(
                weighted_embeddings, token_to_node_flat,
                dim=0, dim_size=num_nodes
            )
            
            # Sum attention weights for normalization
            weight_sum = scatter_sum(
                attention_flat, token_to_node_flat,
                dim=0, dim_size=num_nodes
            )
            
            # Normalize
            node_embeddings_flat = weighted_sum / weight_sum.clamp(min=1e-6)
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        # Reshape back to batch format
        node_embeddings = node_embeddings_flat.view(batch_size, -1, hidden_dim)
        
        return node_embeddings


class NodeToTokenDistributor(nn.Module):
    """
    Distributes node-level representations back to token-level.
    """
    
    def __init__(self, distribution: str = 'copy', combine: str = 'add'):
        """
        Initialize distributor.
        
        Args:
            distribution: How to distribute ('copy', 'learned')
            combine: How to combine with original tokens ('add', 'concat', 'gate')
        """
        super().__init__()
        self.distribution = distribution
        self.combine = combine
        
        if combine == 'gate':
            # Learnable gate for combining token and node features
            self.gate = nn.Linear(1, 1)
    
    def forward(self, node_embeddings: torch.Tensor,
                token_embeddings: torch.Tensor,
                token_to_node: torch.Tensor) -> torch.Tensor:
        """
        Distribute node embeddings back to tokens.
        
        Args:
            node_embeddings: Node representations [batch, num_nodes, hidden_dim]
            token_embeddings: Original token representations [batch, seq_len, hidden_dim]
            token_to_node: Mapping from tokens to nodes [batch, seq_len]
            
        Returns:
            Enhanced token embeddings [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = token_embeddings.shape
        
        # Gather node embeddings for each token
        token_to_node_expanded = token_to_node.unsqueeze(-1).expand(-1, -1, hidden_dim)
        distributed_node_embeddings = torch.gather(
            node_embeddings, 1, token_to_node_expanded
        )
        
        # Combine with original token embeddings
        if self.combine == 'add':
            # Simple addition
            combined = token_embeddings + distributed_node_embeddings
        
        elif self.combine == 'concat':
            # Concatenation (doubles hidden dim)
            combined = torch.cat([token_embeddings, distributed_node_embeddings], dim=-1)
        
        elif self.combine == 'gate':
            # Gated combination
            gate_weight = torch.sigmoid(self.gate(torch.ones(1, device=token_embeddings.device)))
            combined = gate_weight * token_embeddings + (1 - gate_weight) * distributed_node_embeddings
        
        else:
            raise ValueError(f"Unknown combine method: {self.combine}")
        
        return combined


class BatchedScatterOps:
    """
    Efficient batched scatter operations for large-scale processing.
    """
    
    @staticmethod
    def create_node_masks(token_to_node: torch.Tensor, num_nodes: int) -> List[torch.Tensor]:
        """
        Create boolean masks for each node indicating which tokens belong to it.
        
        Args:
            token_to_node: Token to node mapping [batch, seq_len]
            num_nodes: Total number of nodes
            
        Returns:
            List of boolean masks, one per node
        """
        masks = []
        for node_id in range(num_nodes):
            mask = (token_to_node == node_id)
            masks.append(mask)
        return masks
    
    @staticmethod
    def segment_aware_scatter(token_embeddings: torch.Tensor,
                             segment_boundaries: List[Tuple[int, int]],
                             aggregation: str = 'mean') -> torch.Tensor:
        """
        Scatter operation aware of segment boundaries.
        
        Args:
            token_embeddings: Token representations [batch, seq_len, hidden_dim]
            segment_boundaries: List of (start, end) tuples for each segment
            aggregation: Aggregation method
            
        Returns:
            Segment embeddings [batch, num_segments, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = token_embeddings.shape
        num_segments = len(segment_boundaries)
        
        segment_embeddings = torch.zeros(
            batch_size, num_segments, hidden_dim,
            dtype=token_embeddings.dtype,
            device=token_embeddings.device
        )
        
        for seg_idx, (start, end) in enumerate(segment_boundaries):
            segment_tokens = token_embeddings[:, start:end, :]
            
            if aggregation == 'mean':
                segment_embeddings[:, seg_idx, :] = segment_tokens.mean(dim=1)
            elif aggregation == 'sum':
                segment_embeddings[:, seg_idx, :] = segment_tokens.sum(dim=1)
            elif aggregation == 'max':
                segment_embeddings[:, seg_idx, :] = segment_tokens.max(dim=1)[0]
        
        return segment_embeddings


class HierarchicalAggregator(nn.Module):
    """
    Hierarchical aggregation for multi-level graph structures.
    """
    
    def __init__(self, hidden_dim: int, num_levels: int = 3):
        """
        Initialize hierarchical aggregator.
        
        Args:
            hidden_dim: Hidden dimension size
            num_levels: Number of hierarchy levels
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        
        # Level-specific transformations
        self.level_transforms = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_levels)
        ])
        
        # Cross-level attention
        self.cross_level_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )
    
    def forward(self, token_embeddings: torch.Tensor,
                hierarchy: Dict[int, List[List[int]]]) -> Dict[int, torch.Tensor]:
        """
        Perform hierarchical aggregation.
        
        Args:
            token_embeddings: Token representations [batch, seq_len, hidden_dim]
            hierarchy: Dict mapping level -> list of node groupings
            
        Returns:
            Dict mapping level -> aggregated embeddings
        """
        level_embeddings = {}
        previous_level = token_embeddings
        
        for level in range(self.num_levels):
            if level not in hierarchy:
                break
            
            # Get node groupings for this level
            groupings = hierarchy[level]
            num_nodes = len(groupings)
            
            # Aggregate to this level
            level_embeds = []
            for group in groupings:
                # Average tokens/nodes in this group
                group_embed = previous_level[:, group, :].mean(dim=1)
                level_embeds.append(group_embed)
            
            # Stack and transform
            level_embeds = torch.stack(level_embeds, dim=1)
            level_embeds = self.level_transforms[level](level_embeds)
            
            # Apply cross-level attention if not first level
            if level > 0:
                attended, _ = self.cross_level_attention(
                    level_embeds, previous_level, previous_level
                )
                level_embeds = level_embeds + attended
            
            level_embeddings[level] = level_embeds
            previous_level = level_embeds
        
        return level_embeddings