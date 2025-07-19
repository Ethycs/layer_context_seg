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


def build_dual_level_graph(segments: List[Dict], 
                          token_attention: np.ndarray,
                          node_attention: np.ndarray,
                          edge_types: Optional[np.ndarray] = None) -> nx.DiGraph:
    """
    Build a graph with dual-level attention information.
    
    Args:
        segments: List of segment dictionaries
        token_attention: Token-level attention matrix
        node_attention: Node-level attention matrix
        edge_types: Optional edge type matrix
        
    Returns:
        NetworkX graph with dual-level properties
    """
    from .graph_objects import GraphAwareSegment, DualLevelEdge, EdgeType
    
    G = nx.DiGraph()
    
    # Add nodes with GAP properties
    for i, seg in enumerate(segments):
        gap_segment = GraphAwareSegment(
            id=seg['id'],
            content=seg['content'],
            start_pos=seg.get('start_pos', 0),
            end_pos=seg.get('end_pos', len(seg['content'])),
            has_code=seg.get('has_code', False),
            has_math=seg.get('has_math', False),
            cohesion_score=seg.get('cohesion', 0.0),
            node_attention_strength=node_attention[i].mean() if i < node_attention.shape[0] else 0.0
        )
        G.add_node(seg['id'], segment=gap_segment)
    
    # Add dual-level edges
    for i in range(len(segments)):
        for j in range(len(segments)):
            if i != j and node_attention[i, j] > 0.1:  # Threshold
                edge_type_val = edge_types[i, j] if edge_types is not None else 0
                edge_type = EdgeType(int(edge_type_val))
                
                dual_edge = DualLevelEdge(
                    source_id=segments[i]['id'],
                    target_id=segments[j]['id'],
                    edge_type=edge_type,
                    weight=float(node_attention[i, j]),
                    node_level_attention=float(node_attention[i, j]),
                    token_level_attention=float(token_attention[i, j]) if i < token_attention.shape[0] and j < token_attention.shape[1] else 0.0
                )
                
                G.add_edge(
                    segments[i]['id'],
                    segments[j]['id'],
                    dual_edge=dual_edge,
                    weight=dual_edge.combined_strength,
                    edge_type=edge_type.value
                )
    
    return G


def analyze_gap_patterns(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Analyze GAP-specific patterns in the graph.
    
    Args:
        graph: Graph with GAP properties
        
    Returns:
        Analysis results
    """
    analysis = {
        'hub_nodes': [],
        'cohesive_clusters': [],
        'cross_level_consistency': 0.0,
        'attention_flow_patterns': {}
    }
    
    # Find hub nodes (high node attention strength)
    for node_id, data in graph.nodes(data=True):
        segment = data.get('segment')
        if segment and hasattr(segment, 'node_attention_strength'):
            if segment.node_attention_strength > 0.5:  # Threshold
                analysis['hub_nodes'].append({
                    'id': node_id,
                    'strength': segment.node_attention_strength,
                    'cohesion': segment.cohesion_score
                })
    
    # Find cohesive clusters
    cohesive_nodes = [
        node_id for node_id, data in graph.nodes(data=True)
        if data.get('segment') and hasattr(data['segment'], 'cohesion_score')
        and data['segment'].cohesion_score > 0.6
    ]
    
    if cohesive_nodes:
        subgraph = graph.subgraph(cohesive_nodes)
        clusters = list(nx.weakly_connected_components(subgraph))
        analysis['cohesive_clusters'] = [
            list(cluster) for cluster in clusters if len(cluster) > 1
        ]
    
    # Analyze cross-level consistency
    edge_consistencies = []
    for u, v, data in graph.edges(data=True):
        dual_edge = data.get('dual_edge')
        if dual_edge and hasattr(dual_edge, 'cross_level_agreement'):
            edge_consistencies.append(dual_edge.cross_level_agreement)
    
    if edge_consistencies:
        analysis['cross_level_consistency'] = np.mean(edge_consistencies)
    
    return analysis
