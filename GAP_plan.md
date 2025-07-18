# GAP (Graph-Aware Positional) Adaptation Plan for Qwen 2 Transformer

## Overview
This document outlines the complete implementation plan for adapting GAP_COLING2022's graph-aware attention mechanisms to work with the Qwen 2 transformer in the layered-context-graph system. The adaptation preserves and enhances existing compressed attention mechanisms while adding graph-aware capabilities.

## Background
- **GAP**: A framework that enhances language models with graph structural information without extensive pre-training
- **Key Innovation**: Type-aware attention biasing and adjacency-based attention masking
- **Our System**: Uses Qwen 2 transformer with low-rank attention decomposition and windowed processing

## Implementation Plan

### 1. Graph-Aware Attention Module (`graph_aware_attention.py`)

```python
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

from .low_rank_attention import LowRankAttention

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
```

### 2. QwQModel Extensions

Add these methods to `qwq_model.py`:

```python
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
```

### 3. AttentionCalculator Enhancements

Add to `attention_calculator.py`:

```python
def _construct_attention_graph(self, attention_matrix: np.ndarray, 
                             threshold: float = 0.1) -> Tuple[np.ndarray, List[Dict]]:
    """
    Construct a graph from attention patterns.
    
    Args:
        attention_matrix: Averaged attention weights [seq_len, seq_len]
        threshold: Minimum attention weight to create an edge
        
    Returns:
        Tuple of (adjacency_matrix, edge_list)
    """
    seq_len = attention_matrix.shape[0]
    adjacency = np.zeros((seq_len, seq_len))
    edge_list = []
    
    for i in range(seq_len):
        for j in range(seq_len):
            if attention_matrix[i, j] > threshold:
                adjacency[i, j] = 1
                edge_list.append({
                    'from': i,
                    'to': j,
                    'weight': float(attention_matrix[i, j])
                })
    
    return adjacency, edge_list

def process_graph_aware_window(self, window_data: Dict[str, Any]) -> None:
    """
    Process a window with graph-aware attention data.
    
    Args:
        window_data: Window data with graph constraints
    """
    # First run standard processing
    self.process_window(window_data)
    
    if window_data.get('graph_aware', False):
        # Extract graph structure from attention
        for layer in window_data['layers']:
            if layer.get('is_graph_constrained'):
                attention = layer['attention']
                # Average across heads if needed
                if len(attention.shape) > 2:
                    attention = attention.mean(axis=0)
                
                adjacency, edges = self._construct_attention_graph(attention)
                
                # Store graph metrics
                if 'graph_metrics' not in self.window_boundaries[0]:
                    self.window_boundaries[0]['graph_metrics'] = []
                
                self.window_boundaries[0]['graph_metrics'].append({
                    'layer_idx': layer['layer_idx'],
                    'num_edges': len(edges),
                    'graph_density': adjacency.mean(),
                    'strongest_connections': sorted(edges, 
                                                  key=lambda x: x['weight'], 
                                                  reverse=True)[:5]
                })

def get_graph_results(self) -> Dict[str, Any]:
    """
    Get graph-specific results from processed windows.
    """
    results = self.get_results()
    
    # Add graph-specific analysis
    if hasattr(self, 'window_boundaries') and self.window_boundaries:
        graph_data = []
        for window in self.window_boundaries:
            if 'graph_metrics' in window:
                graph_data.extend(window['graph_metrics'])
        
        if graph_data:
            results['graph_analysis'] = {
                'average_density': np.mean([g['graph_density'] for g in graph_data]),
                'total_edges': sum(g['num_edges'] for g in graph_data),
                'layer_statistics': self._compute_layer_graph_stats(graph_data)
            }
    
    return results
```

### 4. Graph Objects Extensions

Add to `graph_objects.py`:

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class EdgeType(Enum):
    """Types of relationships between segments."""
    NO_RELATION = 0
    EXPLAINS = 1
    ELABORATES = 2
    CONTRADICTS = 3
    IS_EXAMPLE_OF = 4
    IS_CONSEQUENCE_OF = 5
    DEPENDS_ON = 6
    SUMMARIZES = 7
    REFERENCES = 8
    CONTINUES = 9

@dataclass
class GraphAwareSegment(EnrichedSegment):
    """Segment with graph-aware properties."""
    incoming_edges: List[Tuple[str, EdgeType]] = field(default_factory=list)
    outgoing_edges: List[Tuple[str, EdgeType]] = field(default_factory=list)
    attention_density: float = 0.0
    type_distribution: Dict[EdgeType, int] = field(default_factory=dict)
    
    def add_incoming_edge(self, from_id: str, edge_type: EdgeType):
        """Add an incoming edge with type."""
        self.incoming_edges.append((from_id, edge_type))
        self.type_distribution[edge_type] = self.type_distribution.get(edge_type, 0) + 1
    
    def add_outgoing_edge(self, to_id: str, edge_type: EdgeType):
        """Add an outgoing edge with type."""
        self.outgoing_edges.append((to_id, edge_type))

@dataclass
class TypeEmbedding:
    """Lightweight type representation for edges."""
    edge_type: EdgeType
    embedding_value: float
    learnable: bool = True
    
    @classmethod
    def create_default_embeddings(cls) -> Dict[EdgeType, 'TypeEmbedding']:
        """Create default type embeddings with small values."""
        default_values = {
            EdgeType.NO_RELATION: 0.0,
            EdgeType.EXPLAINS: 0.05,
            EdgeType.ELABORATES: 0.03,
            EdgeType.CONTRADICTS: -0.1,
            EdgeType.IS_EXAMPLE_OF: 0.04,
            EdgeType.IS_CONSEQUENCE_OF: 0.06,
            EdgeType.DEPENDS_ON: 0.08,
            EdgeType.SUMMARIZES: 0.07,
            EdgeType.REFERENCES: 0.02,
            EdgeType.CONTINUES: 0.01
        }
        return {
            edge_type: cls(edge_type, value) 
            for edge_type, value in default_values.items()
        }
```

### 5. PartitionManager Updates

Add to `partition_manager.py`:

```python
def partition_with_graph_awareness(self, text: str, k_rules: List[str], 
                                 use_attention_graph: bool = True):
    """
    Creates a hierarchical graph with type-aware edges using attention patterns.
    
    Args:
        text: Input text
        k_rules: Segmentation rules
        use_attention_graph: Whether to use attention for graph construction
    """
    # First run standard partitioning
    self.partition(text, k_rules)
    
    if use_attention_graph:
        # Extract attention-based graph structure
        logger.info("Extracting attention-based graph structure...")
        
        # Get attention patterns for the full text
        attention_data = self.segmenter.extract_graph_aware_attention(
            text, rank_ratio=0.1, top_n_layers=4
        )
        
        # Build adjacency from attention
        if attention_data and 'layers' in attention_data[0]:
            self._build_attention_graph(attention_data[0])
        
    # Add type-aware semantic edges
    self._add_typed_semantic_edges()
    
    logger.info(f"Graph-aware partitioning complete with {self.graph.number_of_edges()} edges")

def _build_attention_graph(self, attention_data: Dict[str, Any]):
    """Build graph edges from attention patterns."""
    layers = attention_data.get('layers', [])
    if not layers:
        return
    
    # Average attention across layers
    attention_matrices = [layer['attention'] for layer in layers]
    avg_attention = np.mean(attention_matrices, axis=0)
    
    # If multi-head, average across heads
    if len(avg_attention.shape) > 2:
        avg_attention = avg_attention.mean(axis=0)
    
    # Map attention indices to segments
    tokens = attention_data.get('tokens', [])
    segment_boundaries = self._get_segment_token_boundaries(tokens)
    
    # Create edges based on strong attention
    threshold = 1.0 / len(tokens)  # Above uniform attention
    
    for i, (seg_i_id, seg_i_range) in enumerate(segment_boundaries):
        for j, (seg_j_id, seg_j_range) in enumerate(segment_boundaries):
            if i != j:
                # Aggregate attention between segments
                attention_weight = avg_attention[
                    seg_i_range[0]:seg_i_range[1],
                    seg_j_range[0]:seg_j_range[1]
                ].mean()
                
                if attention_weight > threshold:
                    # Infer edge type
                    edge_type = self._infer_edge_type(
                        self.segments[seg_i_id],
                        self.segments[seg_j_id]
                    )
                    
                    self.graph.add_edge(
                        seg_i_id, seg_j_id,
                        type='attention_derived',
                        weight=float(attention_weight),
                        edge_type=edge_type.value
                    )

def _add_typed_semantic_edges(self):
    """Add semantic edges with type information."""
    if len(self.segments) < 2:
        return
    
    all_segments = list(self.segments.values())
    contents = [seg.content for seg in all_segments]
    embeddings = self.embedding_model.encode(contents, batch_size=32)
    
    # Import type processor
    from models.graph_aware_attention import TypeAwareAttentionProcessor
    type_processor = TypeAwareAttentionProcessor()
    
    for i in range(len(all_segments)):
        for j in range(i + 1, len(all_segments)):
            similarity = np.dot(embeddings[i], embeddings[j])
            if similarity > self.similarity_threshold:
                # Infer edge type
                edge_type = type_processor.infer_edge_type(
                    all_segments[i].content,
                    all_segments[j].content,
                    self.segmenter.classify_relationship
                )
                
                self.graph.add_edge(
                    all_segments[i].id, 
                    all_segments[j].id, 
                    type='semantic_similarity',
                    weight=float(similarity),
                    edge_type=edge_type
                )
                
                # Update segments if using GraphAwareSegment
                if hasattr(all_segments[i], 'add_outgoing_edge'):
                    all_segments[i].add_outgoing_edge(
                        all_segments[j].id, 
                        EdgeType(edge_type)
                    )
                    all_segments[j].add_incoming_edge(
                        all_segments[i].id,
                        EdgeType(edge_type)
                    )

def _infer_edge_type(self, source_segment: EnrichedSegment, 
                    target_segment: EnrichedSegment) -> EdgeType:
    """Infer the edge type between two segments."""
    # Use the segmenter's relationship classifier
    relationship = self.segmenter.classify_relationship(
        source_segment.content,
        target_segment.content
    )
    
    # Map to EdgeType enum
    type_mapping = {
        "explains": EdgeType.EXPLAINS,
        "elaborates": EdgeType.ELABORATES,
        "contradicts": EdgeType.CONTRADICTS,
        "is_example_of": EdgeType.IS_EXAMPLE_OF,
        "is_consequence_of": EdgeType.IS_CONSEQUENCE_OF,
        "depends_on": EdgeType.DEPENDS_ON,
        "no_clear_relation": EdgeType.NO_RELATION
    }
    
    return type_mapping.get(relationship, EdgeType.ELABORATES)
```

### 6. Graph Utilities (`graph_utils.py`)

```python
#!/usr/bin/env python3
"""
Graph Utilities
===============
Utilities for graph construction, analysis, and visualization.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import logging

logger = logging.getLogger(__name__)


def attention_to_adjacency(attention_matrix: np.ndarray, 
                          threshold: float = 0.1,
                          symmetric: bool = False) -> np.ndarray:
    """
    Convert attention matrix to adjacency matrix.
    
    Args:
        attention_matrix: Attention weights [seq_len, seq_len]
        threshold: Minimum weight to create edge
        symmetric: Whether to make adjacency symmetric
        
    Returns:
        Binary adjacency matrix
    """
    adjacency = (attention_matrix > threshold).astype(float)
    
    if symmetric:
        # Make symmetric by taking maximum
        adjacency = np.maximum(adjacency, adjacency.T)
    
    return adjacency


def sparsify_graph(adjacency: np.ndarray, keep_top_k: int = None,
                   keep_threshold: float = None) -> np.ndarray:
    """
    Sparsify graph by keeping only strongest connections.
    
    Args:
        adjacency: Weighted adjacency matrix
        keep_top_k: Keep top-k edges per node
        keep_threshold: Keep edges above threshold
        
    Returns:
        Sparsified adjacency matrix
    """
    sparse_adj = np.zeros_like(adjacency)
    
    if keep_top_k is not None:
        # Keep top-k per row
        for i in range(adjacency.shape[0]):
            top_k_indices = np.argpartition(adjacency[i], -keep_top_k)[-keep_top_k:]
            sparse_adj[i, top_k_indices] = adjacency[i, top_k_indices]
    
    if keep_threshold is not None:
        # Apply threshold
        mask = adjacency > keep_threshold
        sparse_adj = sparse_adj * mask if keep_top_k else adjacency * mask
    
    return sparse_adj


def compute_graph_metrics(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Compute various graph metrics.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'density': nx.density(graph),
        'is_connected': nx.is_weakly_connected(graph),
        'num_components': nx.number_weakly_connected_components(graph)
    }
    
    # Compute centrality measures
    if graph.number_of_nodes() > 0:
        metrics['pagerank'] = nx.pagerank(graph, weight='weight')
        metrics['in_degree_centrality'] = nx.in_degree_centrality(graph)
        metrics['out_degree_centrality'] = nx.out_degree_centrality(graph)
        
        # Find most central nodes
        pagerank_sorted = sorted(metrics['pagerank'].items(), 
                               key=lambda x: x[1], reverse=True)
        metrics['top_nodes'] = pagerank_sorted[:5]
    
    return metrics


def visualize_attention_graph(segments: List[Dict], 
                            adjacency: np.ndarray,
                            edge_types: Optional[np.ndarray] = None,
                            save_path: Optional[str] = None):
    """
    Visualize attention-based graph structure.
    
    Args:
        segments: List of segment dictionaries
        adjacency: Adjacency matrix
        edge_types: Optional edge type matrix
        save_path: Path to save visualization
    """
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes
    for i, segment in enumerate(segments):
        G.add_node(i, label=segment.get('content', '')[:50] + '...')
    
    # Add edges
    for i in range(len(segments)):
        for j in range(len(segments)):
            if adjacency[i, j] > 0:
                edge_attrs = {'weight': adjacency[i, j]}
                if edge_types is not None:
                    edge_attrs['type'] = int(edge_types[i, j])
                G.add_edge(i, j, **edge_attrs)
    
    # Create layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw
    plt.figure(figsize=(12, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1000, alpha=0.8)
    
    # Draw edges with different styles for different types
    if edge_types is not None:
        edge_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for edge_type in range(10):
            edges = [(u, v) for u, v, d in G.edges(data=True) 
                    if d.get('type', 0) == edge_type]
            if edges:
                nx.draw_networkx_edges(G, pos, edgelist=edges,
                                     edge_color=[edge_colors[edge_type]],
                                     width=2, alpha=0.6, 
                                     connectionstyle="arc3,rad=0.1")
    else:
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              width=2, alpha=0.6)
    
    # Draw labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title("Attention-Based Graph Structure")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def analyze_type_distribution(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Analyze distribution of edge types in graph.
    
    Args:
        graph: NetworkX graph with edge type attributes
        
    Returns:
        Type distribution statistics
    """
    type_counts = {}
    type_weights = {}
    
    for u, v, data in graph.edges(data=True):
        edge_type = data.get('edge_type', 0)
        weight = data.get('weight', 1.0)
        
        type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
        type_weights[edge_type] = type_weights.get(edge_type, 0) + weight
    
    # Compute statistics
    total_edges = graph.number_of_edges()
    distribution = {
        'type_counts': type_counts,
        'type_weights': type_weights,
        'type_percentages': {
            t: count / total_edges for t, count in type_counts.items()
        } if total_edges > 0 else {},
        'dominant_type': max(type_counts.items(), key=lambda x: x[1])[0] 
                        if type_counts else None
    }
    
    return distribution


def extract_subgraph_by_type(graph: nx.DiGraph, 
                           edge_types: List[int]) -> nx.DiGraph:
    """
    Extract subgraph containing only specified edge types.
    
    Args:
        graph: Original graph
        edge_types: List of edge types to include
        
    Returns:
        Subgraph with specified edge types
    """
    subgraph = nx.DiGraph()
    
    # Add all nodes
    subgraph.add_nodes_from(graph.nodes(data=True))
    
    # Add edges of specified types
    for u, v, data in graph.edges(data=True):
        if data.get('edge_type', 0) in edge_types:
            subgraph.add_edge(u, v, **data)
    
    return subgraph


def merge_attention_windows(window_graphs: List[Tuple[np.ndarray, int]], 
                          total_length: int,
                          overlap: int) -> np.ndarray:
    """
    Merge attention graphs from overlapping windows.
    
    Args:
        window_graphs: List of (adjacency_matrix, start_position) tuples
        total_length: Total sequence length
        overlap: Overlap between windows
        
    Returns:
        Merged adjacency matrix
    """
    merged = np.zeros((total_length, total_length))
    weights = np.zeros((total_length, total_length))
    
    for adjacency, start_pos in window_graphs:
        window_size = adjacency.shape[0]
        end_pos = min(start_pos + window_size, total_length)
        
        # Add to merged matrix with weights
        merged[start_pos:end_pos, start_pos:end_pos] += adjacency
        weights[start_pos:end_pos, start_pos:end_pos] += 1
    
    # Normalize by weights
    weights[weights == 0] = 1  # Avoid division by zero
    merged = merged / weights
    
    return merged
```

## Integration Example

```python
# Example usage combining all components
from layered_context_graph.src.partitioning.partition_manager import PartitionManager
from layered_context_graph.src.graph.attention_calculator import AttentionCalculator

# Initialize partition manager
pm = PartitionManager()

# Load text
text = "Your document text here..."

# Partition with graph awareness
pm.partition_with_graph_awareness(
    text,
    k_rules=["Segment by paragraph", "Split by topic change"],
    use_attention_graph=True
)

# Extract graph-aware attention patterns
calculator = AttentionCalculator(rank_ratio=0.1, boundary_threshold=0.3)
attention_data = pm.segmenter.extract_graph_aware_attention(
    text,
    adjacency_matrix=pm.graph.to_numpy_array(),
    calculator=calculator
)

# Get results with graph analysis
results = calculator.get_graph_results()

# Classify nodes with type awareness
pm.classify()

# Reassemble with graph-guided synthesis
output = pm.reassemble(
    "Summarize the key concepts and their relationships",
    "graph_aware_summary"
)
```

## Benefits of This Approach

1. **Compression Synergy**: Graph constraints naturally sparsify attention, enhancing low-rank compression
2. **Type-Aware Processing**: Lightweight type embeddings guide attention without heavy computation
3. **Flexible Integration**: Works with existing windowing and streaming approaches
4. **Memory Efficient**: Maintains all benefits of your existing compression infrastructure
5. **Rich Analysis**: Provides both attention patterns and graph structure insights

## Future Enhancements

1. **Learnable Type Embeddings**: Fine-tune type embeddings on specific domains
2. **Dynamic Graph Construction**: Adapt graph structure during processing
3. **Multi-Scale Graphs**: Different granularities for different layers
4. **Graph Neural Network Integration**: Use GNN layers for richer representations
5. **Attention Pattern Mining**: Discover recurring subgraph patterns