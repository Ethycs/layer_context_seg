#!/usr/bin/env python3
"""
PyTorch-based Spectral Clustering for Attention Graphs
=====================================================
GPU-accelerated, differentiable spectral methods for segmenting text
based on real attention patterns from transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class TorchSpectralClustering(nn.Module):
    """
    GPU-accelerated spectral clustering using PyTorch tensors.
    Fully differentiable for end-to-end training.
    """
    
    def __init__(self, 
                 device: str = None,
                 normalized_laplacian: bool = True,
                 epsilon: float = 1e-8):
        """
        Initialize the spectral clustering module.
        
        Args:
            device: Device to run computations on ('cuda', 'cpu', or None for auto)
            normalized_laplacian: Whether to use normalized Laplacian (more stable)
            epsilon: Small value for numerical stability
        """
        super().__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.normalized_laplacian = normalized_laplacian
        self.epsilon = epsilon
        
        logger.info(f"Initialized TorchSpectralClustering on {self.device}")
    
    def forward(self, adjacency_matrix: torch.Tensor, 
                num_clusters: int = 2) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform spectral clustering on the adjacency matrix.
        
        Args:
            adjacency_matrix: NxN adjacency matrix (attention scores between segments)
            num_clusters: Number of clusters to create
            
        Returns:
            Tuple of:
                - cluster_assignments: Tensor of cluster IDs for each node
                - metadata: Dict containing eigenvectors, eigenvalues, etc.
        """
        # Move to correct device if not already there
        if adjacency_matrix.device != self.device:
            A = adjacency_matrix.to(self.device)
        else:
            A = adjacency_matrix
        
        # Ensure symmetry (average with transpose)
        A = (A + A.t()) / 2
        
        # Compute Laplacian
        L = self._compute_laplacian(A)
        
        # Performance optimization for large matrices
        if A.shape[0] > 1000:
            logger.info(f"Large matrix ({A.shape[0]}x{A.shape[0]}), using approximate methods")
            # Use Lanczos algorithm for large matrices (faster)
            eigenvalues, eigenvectors = torch.linalg.eigvalsh(L, eigvals=(0, min(num_clusters + 1, A.shape[0] - 1)))
            k_eigenvectors = eigenvectors[:, :num_clusters]
        else:
            # Compute all eigenvectors for small matrices
            eigenvalues, eigenvectors = torch.linalg.eigh(L)
            k_eigenvectors = eigenvectors[:, :num_clusters]
        
        # Normalize rows for k-means (optional but recommended)
        k_eigenvectors_normalized = F.normalize(k_eigenvectors, p=2, dim=1)
        
        # Simple k-means clustering in spectral space
        # For differentiability, we use soft assignments
        # Reduce iterations for large graphs
        num_iterations = 50 if A.shape[0] > 500 else 100
        cluster_assignments = self._soft_kmeans(k_eigenvectors_normalized, num_clusters, num_iterations)
        
        metadata = {
            'eigenvalues': eigenvalues[:num_clusters],  # Only store what we need
            'eigenvectors': k_eigenvectors,  # Only store k eigenvectors
            'laplacian': L if A.shape[0] < 100 else None,  # Don't store large matrices
            'fiedler_vector': eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else None
        }
        
        return cluster_assignments, metadata
    
    def _compute_laplacian(self, A: torch.Tensor) -> torch.Tensor:
        """
        Compute the graph Laplacian.
        
        Args:
            A: Adjacency matrix
            
        Returns:
            L: Laplacian matrix
        """
        # Degree matrix
        D = torch.diag(A.sum(dim=1))
        
        if self.normalized_laplacian:
            # Normalized Laplacian: L = I - D^(-1/2) @ A @ D^(-1/2)
            D_sqrt_inv = torch.diag(1.0 / (torch.sqrt(D.diagonal()) + self.epsilon))
            L = torch.eye(A.shape[0], device=self.device) - D_sqrt_inv @ A @ D_sqrt_inv
        else:
            # Unnormalized Laplacian: L = D - A
            L = D - A
            
        return L
    
    def _soft_kmeans(self, X: torch.Tensor, k: int, 
                     num_iterations: int = 100) -> torch.Tensor:
        """
        Differentiable soft k-means clustering with memory optimization.
        
        Args:
            X: Data points to cluster (N x D)
            k: Number of clusters
            num_iterations: Number of iterations
            
        Returns:
            Soft cluster assignments (N x k)
        """
        N, D = X.shape
        
        # Check GPU memory for large datasets
        if self.device.type == 'cuda' and N > 10000:
            # Estimate memory usage
            mem_required = N * k * 4 * 2  # float32, distances and assignments
            mem_available = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            
            if mem_required > mem_available * 0.8:  # Use only 80% of available memory
                logger.warning(f"Large dataset ({N} points), using mini-batch k-means")
                return self._minibatch_soft_kmeans(X, k, num_iterations)
        
        # Initialize centroids using k-means++
        centroids = self._initialize_centroids_plusplus(X, k)
        
        for _ in range(num_iterations):
            # Compute distances to centroids
            distances = torch.cdist(X, centroids)
            
            # Soft assignments using softmax (differentiable)
            soft_assignments = F.softmax(-distances, dim=1)
            
            # Update centroids
            centroids = soft_assignments.t() @ X
            centroids = centroids / (soft_assignments.sum(dim=0, keepdim=True).t() + self.epsilon)
        
        return soft_assignments
    
    def _minibatch_soft_kmeans(self, X: torch.Tensor, k: int, 
                               num_iterations: int = 100, 
                               batch_size: int = 1024) -> torch.Tensor:
        """
        Mini-batch version of soft k-means for memory efficiency.
        """
        N, D = X.shape
        
        # Initialize centroids
        centroids = self._initialize_centroids_plusplus(X, k)
        
        # Process in batches
        soft_assignments = torch.zeros((N, k), device=self.device)
        
        for _ in range(num_iterations):
            # Reset accumulator for centroids
            new_centroids = torch.zeros_like(centroids)
            total_weights = torch.zeros(k, device=self.device)
            
            # Process each batch
            for i in range(0, N, batch_size):
                batch = X[i:min(i + batch_size, N)]
                
                # Compute distances for this batch
                distances = torch.cdist(batch, centroids)
                batch_assignments = F.softmax(-distances, dim=1)
                
                # Store assignments
                soft_assignments[i:min(i + batch_size, N)] = batch_assignments
                
                # Accumulate for centroid update
                new_centroids += batch_assignments.t() @ batch
                total_weights += batch_assignments.sum(dim=0)
            
            # Update centroids
            centroids = new_centroids / (total_weights.unsqueeze(1) + self.epsilon)
        
        return soft_assignments
    
    def _initialize_centroids_plusplus(self, X: torch.Tensor, k: int) -> torch.Tensor:
        """
        Initialize centroids using k-means++ algorithm.
        
        Args:
            X: Data points (N x D)
            k: Number of centroids
            
        Returns:
            Initial centroids (k x D)
        """
        N, D = X.shape
        centroids = torch.zeros((k, D), device=self.device)
        
        # Choose first centroid randomly
        first_idx = torch.randint(0, N, (1,)).item()
        centroids[0] = X[first_idx]
        
        for i in range(1, k):
            # Compute distances to existing centroids
            distances = torch.cdist(X, centroids[:i])
            min_distances = distances.min(dim=1)[0]
            
            # Choose next centroid with probability proportional to squared distance
            probabilities = min_distances ** 2
            probabilities = probabilities / probabilities.sum()
            
            # Sample next centroid
            next_idx = torch.multinomial(probabilities, 1).item()
            centroids[i] = X[next_idx]
        
        return centroids
    
    def hierarchical_partition(self, adjacency_matrix: torch.Tensor,
                              max_depth: int = 3,
                              min_cluster_size: int = 2) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Perform hierarchical spectral partitioning using recursive Fiedler vector splits.
        
        Args:
            adjacency_matrix: NxN adjacency matrix
            max_depth: Maximum depth of hierarchy
            min_cluster_size: Minimum size for a cluster to be split further
            
        Returns:
            Dictionary containing hierarchical clustering results
        """
        A = adjacency_matrix.to(self.device)
        N = A.shape[0]
        
        # Initialize cluster hierarchy
        hierarchy = {
            'levels': [],
            'tree': {},
            'final_assignments': torch.zeros(N, dtype=torch.long, device=self.device)
        }
        
        # Recursive partitioning
        self._recursive_partition(
            A, 
            torch.arange(N, device=self.device),
            hierarchy,
            depth=0,
            max_depth=max_depth,
            min_cluster_size=min_cluster_size,
            cluster_id=0
        )
        
        return hierarchy
    
    def _recursive_partition(self, A: torch.Tensor, indices: torch.Tensor,
                           hierarchy: Dict, depth: int, max_depth: int,
                           min_cluster_size: int, cluster_id: int):
        """
        Recursively partition using Fiedler vector.
        """
        if depth >= max_depth or len(indices) < min_cluster_size * 2:
            # Base case: assign all nodes to current cluster
            hierarchy['final_assignments'][indices] = cluster_id
            return cluster_id + 1
        
        # Extract subgraph
        subgraph = A[indices][:, indices]
        
        # Compute Laplacian and Fiedler vector
        L = self._compute_laplacian(subgraph)
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        
        # Fiedler vector is the eigenvector of the second smallest eigenvalue
        fiedler = eigenvectors[:, 1]
        
        # Partition based on sign of Fiedler vector
        partition1 = indices[fiedler >= 0]
        partition2 = indices[fiedler < 0]
        
        # Record this level in hierarchy
        if depth not in hierarchy['tree']:
            hierarchy['tree'][depth] = []
        
        hierarchy['tree'][depth].append({
            'indices': indices,
            'partition1': partition1,
            'partition2': partition2,
            'fiedler': fiedler
        })
        
        # Recursive calls
        next_cluster_id = cluster_id
        if len(partition1) >= min_cluster_size:
            next_cluster_id = self._recursive_partition(
                A, partition1, hierarchy, depth + 1, max_depth, 
                min_cluster_size, next_cluster_id
            )
        else:
            hierarchy['final_assignments'][partition1] = next_cluster_id
            next_cluster_id += 1
            
        if len(partition2) >= min_cluster_size:
            next_cluster_id = self._recursive_partition(
                A, partition2, hierarchy, depth + 1, max_depth,
                min_cluster_size, next_cluster_id
            )
        else:
            hierarchy['final_assignments'][partition2] = next_cluster_id
            next_cluster_id += 1
            
        return next_cluster_id


class ConversationAwareSpectralClustering(TorchSpectralClustering):
    """
    Spectral clustering with special handling for conversation structure.
    """
    
    def __init__(self, 
                 speaker_boundary_weight: float = 2.0,
                 temporal_decay: float = 0.95,
                 reference_amplification: float = 1.5,
                 **kwargs):
        """
        Initialize conversation-aware spectral clustering.
        
        Args:
            speaker_boundary_weight: Weight multiplier for speaker boundaries
            temporal_decay: Decay factor for long-range attention
            reference_amplification: Amplification for reference connections
        """
        super().__init__(**kwargs)
        
        self.speaker_boundary_weight = speaker_boundary_weight
        self.temporal_decay = temporal_decay
        self.reference_amplification = reference_amplification
    
    def preprocess_attention_for_conversation(self, 
                                            attention_matrix: torch.Tensor,
                                            speaker_labels: Optional[torch.Tensor] = None,
                                            reference_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Preprocess attention matrix with conversation-specific biases.
        
        Args:
            attention_matrix: Raw attention scores
            speaker_labels: Tensor indicating speaker for each segment
            reference_mask: Boolean mask for reference connections
            
        Returns:
            Biased attention matrix optimized for conversation segmentation
        """
        A = attention_matrix.clone()
        N = A.shape[0]
        
        # Apply temporal decay for long-range connections
        positions = torch.arange(N, device=self.device).float()
        distance_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        temporal_weights = torch.pow(self.temporal_decay, distance_matrix)
        A = A * temporal_weights
        
        # Boost speaker boundaries
        if speaker_labels is not None:
            # Create speaker change mask
            speaker_change = speaker_labels.unsqueeze(0) != speaker_labels.unsqueeze(1)
            # Reduce attention across speaker boundaries
            A[speaker_change] = A[speaker_change] / self.speaker_boundary_weight
        
        # Amplify reference connections
        if reference_mask is not None:
            A[reference_mask] = A[reference_mask] * self.reference_amplification
        
        return A
    
    def segment_conversation(self, 
                           attention_matrix: torch.Tensor,
                           speaker_labels: Optional[torch.Tensor] = None,
                           mode: str = 'timeline') -> Dict[str, torch.Tensor]:
        """
        Segment conversation using mode-specific spectral clustering.
        
        Args:
            attention_matrix: Attention scores between conversation segments
            speaker_labels: Speaker ID for each segment
            mode: Segmentation mode ('timeline', 'speaker', 'evolution', 'topics')
            
        Returns:
            Segmentation results with cluster assignments and metadata
        """
        # Preprocess attention based on mode
        if mode == 'speaker':
            # Strong bias for speaker boundaries
            A = self.preprocess_attention_for_conversation(
                attention_matrix, 
                speaker_labels=speaker_labels
            )
            # Use hierarchical clustering to respect speaker structure
            results = self.hierarchical_partition(A, max_depth=2)
            
        elif mode == 'evolution':
            # Focus on how ideas develop over time
            A = attention_matrix  # Use raw attention to track idea flow
            # Soft clustering to capture overlapping concepts
            assignments, metadata = self.forward(A, num_clusters=5)
            results = {
                'assignments': assignments,
                'evolution_flow': self._trace_concept_evolution(A, assignments),
                **metadata
            }
            
        elif mode == 'topics':
            # Topic-based clustering with less temporal bias
            A = self.preprocess_attention_for_conversation(
                attention_matrix,
                speaker_labels=speaker_labels
            )
            # Standard spectral clustering
            assignments, metadata = self.forward(A, num_clusters=4)
            results = {'assignments': assignments, **metadata}
            
        else:  # timeline
            # Preserve temporal order while finding natural breaks
            A = self.preprocess_attention_for_conversation(
                attention_matrix,
                speaker_labels=speaker_labels
            )
            # Use Fiedler vector for binary splits at natural boundaries
            results = self._timeline_segmentation(A)
        
        return results
    
    def _timeline_segmentation(self, A: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Segment preserving timeline using recursive Fiedler cuts.
        """
        # Compute Laplacian
        L = self._compute_laplacian(A)
        
        # Get Fiedler vector
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        fiedler = eigenvectors[:, 1]
        
        # Find optimal cut point that preserves order
        # Look for largest gap in Fiedler vector values
        sorted_fiedler, sorted_indices = torch.sort(fiedler)
        gaps = sorted_fiedler[1:] - sorted_fiedler[:-1]
        optimal_cut_idx = gaps.argmax() + 1
        
        # Create binary segmentation
        assignments = torch.zeros_like(fiedler, dtype=torch.long)
        assignments[sorted_indices[optimal_cut_idx:]] = 1
        
        return {
            'assignments': assignments,
            'fiedler_vector': fiedler,
            'cut_point': optimal_cut_idx,
            'eigenvalues': eigenvalues
        }
    
    def _trace_concept_evolution(self, A: torch.Tensor, 
                                assignments: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Trace how concepts evolve through the conversation.
        """
        # Build concept flow graph
        num_clusters = assignments.shape[1]
        concept_flow = torch.zeros((num_clusters, num_clusters), device=self.device)
        
        # Aggregate attention between consecutive segments of different clusters
        for i in range(len(assignments) - 1):
            curr_probs = assignments[i]
            next_probs = assignments[i + 1]
            
            # Weighted flow between concepts
            flow = torch.outer(curr_probs, next_probs) * A[i, i+1]
            concept_flow += flow
        
        return {
            'concept_flow_matrix': concept_flow,
            'dominant_flow': concept_flow.argmax(dim=1)
        }


def attention_to_adjacency(attention_tensor: torch.Tensor,
                          aggregation: str = 'mean',
                          symmetric: bool = True) -> torch.Tensor:
    """
    Convert multi-head attention tensor to adjacency matrix.
    
    Args:
        attention_tensor: Attention weights (layers, heads, seq, seq) or (heads, seq, seq)
        aggregation: How to aggregate across heads/layers ('mean', 'max', 'weighted')
        symmetric: Whether to make the adjacency matrix symmetric
        
    Returns:
        Adjacency matrix (seq, seq)
    """
    # Handle different tensor shapes
    if attention_tensor.dim() == 4:  # (layers, heads, seq, seq)
        # Aggregate across layers and heads
        if aggregation == 'mean':
            A = attention_tensor.mean(dim=(0, 1))
        elif aggregation == 'max':
            A = attention_tensor.max(dim=0)[0].max(dim=0)[0]
        else:  # weighted
            # Later layers and certain heads might be more important
            layer_weights = torch.linspace(0.5, 1.0, attention_tensor.shape[0])
            layer_weights = layer_weights / layer_weights.sum()
            A = (attention_tensor * layer_weights.view(-1, 1, 1, 1)).sum(dim=0).mean(dim=0)
    elif attention_tensor.dim() == 3:  # (heads, seq, seq)
        if aggregation == 'mean':
            A = attention_tensor.mean(dim=0)
        else:
            A = attention_tensor.max(dim=0)[0]
    else:
        A = attention_tensor
    
    # Make symmetric if requested
    if symmetric:
        A = (A + A.t()) / 2
    
    return A