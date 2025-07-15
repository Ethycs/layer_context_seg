#!/usr/bin/env python3
"""
PyTorch Attention Graph Builder
==============================
End-to-end tensor processing for building graphs from transformer attention.
Integrates with spectral clustering for GPU-accelerated segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

from .torch_spectral_clustering import TorchSpectralClustering, ConversationAwareSpectralClustering

logger = logging.getLogger(__name__)


class TorchAttentionGraphBuilder(nn.Module):
    """
    Builds knowledge graphs directly from transformer attention tensors.
    Fully differentiable and GPU-accelerated.
    """
    
    def __init__(self, 
                 device: str = None,
                 spectral_clustering: bool = True,
                 clustering_method: str = 'spectral'):
        """
        Initialize the attention graph builder.
        
        Args:
            device: Device for computation
            spectral_clustering: Whether to use spectral clustering
            clustering_method: Clustering method ('spectral', 'kmeans', 'hierarchical')
        """
        super().__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.spectral_clustering = spectral_clustering
        self.clustering_method = clustering_method
        
        # Initialize spectral clustering module
        if spectral_clustering:
            self.spectral = TorchSpectralClustering(device=self.device)
            self.conv_spectral = ConversationAwareSpectralClustering(device=self.device)
        
        logger.info(f"Initialized TorchAttentionGraphBuilder on {self.device}")
    
    def forward(self, 
                attention_tensor: torch.Tensor,
                text_segments: List[str],
                segment_embeddings: Optional[torch.Tensor] = None,
                instruction_bias: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Build graph from attention tensor.
        
        Args:
            attention_tensor: Attention weights from transformer
                             Shape: (layers, heads, seq, seq) or (heads, seq, seq)
            text_segments: List of text segments corresponding to sequence positions
            segment_embeddings: Optional embeddings for each segment
            instruction_bias: Optional bias tensor from linguistic programming
            
        Returns:
            Graph dictionary with nodes, edges, and clustering results
        """
        # Convert attention to adjacency matrix
        adjacency = self.attention_to_adjacency(attention_tensor)
        
        # Apply instruction bias if provided
        if instruction_bias is not None:
            adjacency = self._apply_instruction_bias(adjacency, instruction_bias)
        
        # Create nodes
        nodes = self._create_nodes(text_segments, segment_embeddings)
        
        # Perform clustering if enabled
        clustering_results = None
        if self.spectral_clustering:
            clustering_results = self.spectral.forward(adjacency)
            nodes = self._assign_clusters_to_nodes(nodes, clustering_results[0])
        
        # Create edges from adjacency matrix
        edges = self._create_edges_from_adjacency(adjacency, threshold=0.1)
        
        # Build final graph structure
        graph = {
            'nodes': nodes,
            'edges': edges,
            'adjacency_matrix': adjacency,
            'clustering': clustering_results,
            'device': str(self.device),
            'method': self.clustering_method
        }
        
        return graph
    
    def attention_to_adjacency(self, 
                              attention_tensor: torch.Tensor,
                              aggregation: str = 'mean') -> torch.Tensor:
        """
        Convert attention tensor to adjacency matrix.
        
        Args:
            attention_tensor: Multi-dimensional attention tensor
            aggregation: How to aggregate ('mean', 'max', 'weighted')
            
        Returns:
            Adjacency matrix (2D)
        """
        # Move to device
        attention = attention_tensor.to(self.device)
        
        # Log tensor shape for debugging
        logger.debug(f"Attention tensor shape: {attention.shape}, dims: {attention.dim()}")
        
        # Handle different shapes
        if attention.dim() == 5:  # (batch, layers, heads, seq, seq) or similar
            # Aggregate over batch dimension first if present
            if attention.shape[0] == 1:
                # Single batch, squeeze it out
                attention = attention.squeeze(0)
                adjacency = self.attention_to_adjacency(attention, aggregation)
            else:
                # Multiple batches, average them
                adjacency = attention.mean(dim=0)
                adjacency = self.attention_to_adjacency(adjacency, aggregation)
            return adjacency
            
        elif attention.dim() == 4:  # (layers, heads, seq, seq)
            if aggregation == 'weighted':
                # Weight later layers more heavily
                layer_weights = torch.linspace(0.3, 1.0, attention.shape[0], device=self.device)
                layer_weights = F.softmax(layer_weights, dim=0)
                weighted = attention * layer_weights.view(-1, 1, 1, 1)
                adjacency = weighted.sum(dim=0).mean(dim=0)
            else:
                adjacency = attention.mean(dim=(0, 1))
                
        elif attention.dim() == 3:  # (heads, seq, seq)
            adjacency = attention.mean(dim=0)
            
        elif attention.dim() == 2:  # Already 2D
            adjacency = attention
            
        else:
            logger.warning(f"Unexpected attention tensor dimensions: {attention.shape}")
            # Try to reduce to 2D by averaging all but last 2 dimensions
            while attention.dim() > 2:
                attention = attention.mean(dim=0)
            adjacency = attention
        
        # Ensure we have a 2D tensor
        if adjacency.dim() != 2:
            logger.error(f"Adjacency matrix has wrong dimensions: {adjacency.shape}")
            raise ValueError(f"Expected 2D adjacency matrix, got {adjacency.dim()}D")
        
        # Make symmetric
        adjacency = (adjacency + adjacency.t()) / 2
        
        return adjacency
    
    def _apply_instruction_bias(self, 
                               adjacency: torch.Tensor,
                               instruction_bias: torch.Tensor) -> torch.Tensor:
        """
        Apply linguistic programming bias to adjacency matrix.
        
        Args:
            adjacency: Base adjacency matrix
            instruction_bias: Bias tensor from instruction seeding
            
        Returns:
            Biased adjacency matrix
        """
        # Ensure same device
        bias = instruction_bias.to(self.device)
        
        # Apply bias (multiplicative or additive)
        if bias.shape == adjacency.shape:
            # Full bias matrix
            biased = adjacency * (1 + bias)
        else:
            # Diagonal bias (affects self-connections)
            biased = adjacency.clone()
            biased.diagonal().add_(bias)
        
        # Ensure non-negative
        biased = F.relu(biased)
        
        return biased
    
    def _create_nodes(self, 
                     text_segments: List[str],
                     embeddings: Optional[torch.Tensor] = None) -> List[Dict[str, Any]]:
        """
        Create node dictionaries from text segments.
        """
        nodes = []
        
        for i, segment in enumerate(text_segments):
            node = {
                'id': f'node_{i}',
                'content': segment,
                'index': i,
                'features': {}
            }
            
            # Add embedding if available
            if embeddings is not None and i < embeddings.shape[0]:
                node['embedding'] = embeddings[i].cpu().numpy()
            
            # Extract basic features
            node['features']['length'] = len(segment.split())
            node['features']['has_question'] = '?' in segment
            node['features']['has_code'] = any(marker in segment for marker in ['```', 'def', 'class'])
            
            nodes.append(node)
        
        return nodes
    
    def _assign_clusters_to_nodes(self, 
                                 nodes: List[Dict],
                                 cluster_assignments: torch.Tensor) -> List[Dict]:
        """
        Add cluster assignments to nodes.
        """
        if cluster_assignments.dim() == 2:  # Soft assignments
            # Use argmax for hard assignment
            hard_assignments = cluster_assignments.argmax(dim=1)
            for i, node in enumerate(nodes):
                if i < len(hard_assignments):
                    node['cluster'] = int(hard_assignments[i])
                    node['cluster_probs'] = cluster_assignments[i].cpu().numpy()
        else:  # Hard assignments
            for i, node in enumerate(nodes):
                if i < len(cluster_assignments):
                    node['cluster'] = int(cluster_assignments[i])
        
        return nodes
    
    def _create_edges_from_adjacency(self, 
                                    adjacency: torch.Tensor,
                                    threshold: float = 0.1,
                                    use_sparse: bool = True,
                                    top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Create edge list from adjacency matrix with sparse optimization.
        
        Args:
            adjacency: Adjacency matrix
            threshold: Minimum weight threshold
            use_sparse: Use sparse operations for large graphs
            top_k: Keep only top-k edges per node
        """
        edges = []
        n_nodes = adjacency.shape[0]
        
        # Auto-enable pruning for large graphs
        if n_nodes > 100 and top_k is None:
            top_k = min(20, n_nodes // 5)
            logger.info(f"Large graph ({n_nodes} nodes), keeping top {top_k} edges per node")
        
        # Apply top-k pruning if requested
        if top_k is not None:
            # Keep top-k connections per node
            topk_values, topk_indices = torch.topk(adjacency, min(top_k, n_nodes), dim=1)
            
            # Create sparse adjacency with only top-k edges
            sparse_adj = torch.zeros_like(adjacency)
            for i in range(n_nodes):
                sparse_adj[i, topk_indices[i]] = topk_values[i]
            
            # Make symmetric by taking max
            adjacency = torch.max(sparse_adj, sparse_adj.t())
        
        # Apply threshold
        if use_sparse and n_nodes > 50:
            # Use sparse operations for efficiency
            mask = adjacency > threshold
            
            # Remove self-loops
            mask.fill_diagonal_(False)
            
            # Get indices efficiently
            indices = mask.nonzero(as_tuple=False)
            
            # Create edges (avoid duplicates by only taking upper triangle)
            seen = set()
            for idx in indices:
                i, j = idx[0].item(), idx[1].item()
                
                # Create canonical edge (smaller index first)
                edge_key = (min(i, j), max(i, j))
                if edge_key not in seen:
                    seen.add(edge_key)
                    
                    edge = {
                        'source': f'node_{i}',
                        'target': f'node_{j}',
                        'weight': float(adjacency[i, j]),
                        'source_idx': i,
                        'target_idx': j
                    }
                    edges.append(edge)
        else:
            # Original implementation for small graphs
            indices = torch.nonzero(adjacency > threshold, as_tuple=False)
            
            for idx in indices:
                i, j = idx[0].item(), idx[1].item()
                
                # Skip self-loops and duplicates
                if i >= j:
                    continue
                
                edge = {
                    'source': f'node_{i}',
                    'target': f'node_{j}',
                    'weight': float(adjacency[i, j]),
                    'source_idx': i,
                    'target_idx': j
                }
                edges.append(edge)
        
        # Add edge type classification for important edges
        if len(edges) > 0:
            # Sort by weight and mark top edges
            edges.sort(key=lambda e: e['weight'], reverse=True)
            
            # Mark top 10% as strong connections
            num_strong = max(1, len(edges) // 10)
            for i, edge in enumerate(edges):
                if i < num_strong:
                    edge['connection_strength'] = 'strong'
                elif i < len(edges) // 3:
                    edge['connection_strength'] = 'medium'
                else:
                    edge['connection_strength'] = 'weak'
        
        logger.info(f"Max adjacency value: {adjacency.max().item()}")
        logger.info(f"Created {len(edges)} edges from {n_nodes}x{n_nodes} adjacency matrix with threshold {threshold}")
        return edges
    
    def build_conversation_graph(self,
                               attention_tensor: torch.Tensor,
                               text_segments: List[str],
                               speaker_labels: Optional[List[str]] = None,
                               mode: str = 'timeline') -> Dict[str, Any]:
        """
        Build conversation-specific graph with appropriate clustering.
        
        Args:
            attention_tensor: Attention from transformer
            text_segments: Conversation segments
            speaker_labels: Speaker for each segment
            mode: Conversation mode ('timeline', 'speaker', 'evolution')
            
        Returns:
            Conversation graph with mode-specific clustering
        """
        # Convert attention to adjacency
        adjacency = self.attention_to_adjacency(attention_tensor)
        
        # Convert speaker labels to tensor if provided
        speaker_tensor = None
        if speaker_labels:
            # Create numeric labels
            unique_speakers = list(set(speaker_labels))
            speaker_map = {speaker: i for i, speaker in enumerate(unique_speakers)}
            speaker_tensor = torch.tensor(
                [speaker_map[s] for s in speaker_labels],
                device=self.device
            )
        
        # Create nodes with speaker info
        nodes = self._create_nodes(text_segments)
        if speaker_labels:
            for node, speaker in zip(nodes, speaker_labels):
                node['speaker'] = speaker
        
        # Perform conversation-aware clustering
        clustering_results = self.conv_spectral.segment_conversation(
            adjacency,
            speaker_labels=speaker_tensor,
            mode=mode
        )
        
        # Assign clusters to nodes
        if 'assignments' in clustering_results:
            nodes = self._assign_clusters_to_nodes(nodes, clustering_results['assignments'])
        
        # Create edges with conversation-specific types
        edges = self._create_conversation_edges(adjacency, nodes, speaker_labels)
        
        # Build graph
        graph = {
            'nodes': nodes,
            'edges': edges,
            'adjacency_matrix': adjacency,
            'clustering': clustering_results,
            'mode': mode,
            'speakers': unique_speakers if speaker_labels else None
        }
        
        return graph
    
    def _create_conversation_edges(self,
                                  adjacency: torch.Tensor,
                                  nodes: List[Dict],
                                  speaker_labels: Optional[List[str]] = None) -> List[Dict]:
        """
        Create edges with conversation-specific types.
        """
        edges = []
        threshold = 0.1
        
        # Get significant connections
        indices = torch.nonzero(adjacency > threshold, as_tuple=False)
        
        for idx in indices:
            i, j = idx[0].item(), idx[1].item()
            
            if i >= j:  # Skip self-loops and duplicates
                continue
            
            # Determine edge type
            edge_type = 'connects'
            if speaker_labels and i < len(speaker_labels) and j < len(speaker_labels):
                if abs(j - i) == 1:  # Adjacent segments
                    if speaker_labels[i] == speaker_labels[j]:
                        edge_type = 'continues'
                    else:
                        edge_type = 'responds_to'
                elif speaker_labels[i] != speaker_labels[j]:
                    edge_type = 'references'
            
            edge = {
                'source': f'node_{i}',
                'target': f'node_{j}',
                'weight': float(adjacency[i, j]),
                'type': edge_type,
                'source_idx': i,
                'target_idx': j
            }
            
            edges.append(edge)
        
        return edges


class DifferentiableConversationSegmenter(nn.Module):
    """
    Fully differentiable conversation segmentation pipeline.
    Can be trained end-to-end to improve segmentation quality.
    """
    
    def __init__(self,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 device: str = None):
        """
        Initialize the differentiable segmenter.
        
        Args:
            hidden_dim: Hidden dimension for learnable parameters
            num_heads: Number of attention heads to specialize
            device: Computation device
        """
        super().__init__()
        
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Learnable parameters for attention head specialization
        self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
        
        # Learnable instruction embeddings
        self.instruction_embeddings = nn.Embedding(10, hidden_dim)  # 10 instruction types
        
        # Attention bias predictor
        self.bias_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Graph builder and spectral clustering
        self.graph_builder = TorchAttentionGraphBuilder(device=self.device)
        
    def forward(self,
                attention_tensor: torch.Tensor,
                segment_embeddings: torch.Tensor,
                instruction_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        End-to-end differentiable segmentation.
        
        Args:
            attention_tensor: Attention from transformer (heads, seq, seq)
            segment_embeddings: Embeddings for each segment (seq, hidden)
            instruction_ids: Optional instruction type IDs
            
        Returns:
            Segmentation results with gradients preserved
        """
        batch_size, seq_len = segment_embeddings.shape[:2]
        
        # Weight attention heads based on learned specialization
        weighted_attention = attention_tensor * self.head_weights.view(-1, 1, 1)
        adjacency = weighted_attention.sum(dim=0)
        
        # Apply instruction bias if provided
        if instruction_ids is not None:
            # Get instruction embeddings
            inst_emb = self.instruction_embeddings(instruction_ids)
            
            # Predict bias for each segment pair
            seg_pairs = torch.cartesian_prod(
                torch.arange(seq_len, device=self.device),
                torch.arange(seq_len, device=self.device)
            )
            
            pair_features = segment_embeddings[seg_pairs[:, 0]] + segment_embeddings[seg_pairs[:, 1]]
            pair_features = pair_features + inst_emb.mean(dim=0, keepdim=True)
            
            bias_values = self.bias_predictor(pair_features).squeeze(-1)
            bias_matrix = bias_values.view(seq_len, seq_len)
            
            # Apply bias to adjacency
            adjacency = adjacency * (1 + bias_matrix)
        
        # Normalize adjacency
        adjacency = F.softmax(adjacency.view(-1), dim=0).view(seq_len, seq_len)
        
        # Make symmetric
        adjacency = (adjacency + adjacency.t()) / 2
        
        # Perform differentiable spectral clustering
        # Using soft k-means in spectral space
        L = torch.eye(seq_len, device=self.device) - adjacency
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        
        # Use first k eigenvectors (k=3 for example)
        k = min(3, seq_len)
        spectral_features = eigenvectors[:, :k]
        
        # Soft clustering assignments (differentiable)
        cluster_logits = spectral_features @ spectral_features.t()
        cluster_probs = F.softmax(cluster_logits, dim=1)
        
        return {
            'adjacency': adjacency,
            'cluster_probs': cluster_probs,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'head_weights': self.head_weights
        }
    
    def segment_loss(self, 
                    cluster_probs: torch.Tensor,
                    target_segments: torch.Tensor,
                    adjacency: torch.Tensor) -> torch.Tensor:
        """
        Compute segmentation loss for training.
        
        Args:
            cluster_probs: Predicted cluster probabilities
            target_segments: Ground truth segment labels
            adjacency: Predicted adjacency matrix
            
        Returns:
            Scalar loss value
        """
        # Clustering loss (cross-entropy with soft targets)
        cluster_loss = F.cross_entropy(
            cluster_probs,
            target_segments
        )
        
        # Graph regularization (encourage block-diagonal structure)
        # Segments should have high within-cluster connectivity
        segment_mask = target_segments.unsqueeze(0) == target_segments.unsqueeze(1)
        within_cluster_adj = adjacency * segment_mask.float()
        between_cluster_adj = adjacency * (~segment_mask).float()
        
        # Maximize within-cluster, minimize between-cluster
        graph_loss = -within_cluster_adj.mean() + between_cluster_adj.mean()
        
        # Total loss
        total_loss = cluster_loss + 0.1 * graph_loss
        
        return total_loss
