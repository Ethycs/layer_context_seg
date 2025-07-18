#!/usr/bin/env python3
"""
Attention Calculator
====================
This module provides an attention calculator that processes attention windows
efficiently with low-rank decomposition support and boundary detection.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
import torch

logger = logging.getLogger(__name__)


class AttentionCalculator:
    """
    A stateful calculator for processing attention windows with low-rank support.
    Compatible with the windowing approach used by QwQModel and PartitionManager.
    """
    
    def __init__(self, rank_ratio: float = 0.1, top_n_layers: int = 4,
                 boundary_threshold: float = 0.3, min_segment_length: int = 50,
                 node_attention_weight: float = 0.5):
        """
        Initialize the attention calculator.
        
        Args:
            rank_ratio: Compression ratio for low-rank decomposition
            top_n_layers: Number of top layers to process
            boundary_threshold: Threshold for detecting segment boundaries
            min_segment_length: Minimum characters per segment
            node_attention_weight: Weight for node-level attention in analysis
        """
        self.rank_ratio = rank_ratio
        self.top_n_layers = top_n_layers
        self.boundary_threshold = boundary_threshold
        self.min_segment_length = min_segment_length
        self.node_attention_weight = node_attention_weight
        
        # State for accumulating results
        self.window_boundaries = []
        self.attention_patterns = []
        self.token_positions = []
        self.aggregate_attention_flow = None
        
        # Additional state for dual-level processing
        self.node_patterns = []
        self.cross_level_patterns = []
        self.segment_cohesion_scores = []
        self.dual_level_boundaries = []
        
    def process_window(self, window_data: Dict[str, Any]) -> None:
        """
        Process a single window of attention data.
        
        Args:
            window_data: Dictionary containing:
                - layers: List of layer attention data
                - tokens: List of tokens
                - metadata: Window metadata
                - low_rank: Whether data is compressed
        """
        if not window_data or 'layers' not in window_data:
            return
            
        # Extract window metadata
        metadata = window_data.get('metadata', {})
        window_idx = metadata.get('window_index', 0)
        token_start = metadata.get('token_start_index', 0)
        
        # Process attention patterns for boundary detection
        boundaries = self._detect_boundaries_in_window(
            window_data['layers'],
            window_data.get('tokens', []),
            token_start
        )
        
        # Store boundaries with window context
        for boundary in boundaries:
            self.window_boundaries.append({
                'window_index': window_idx,
                'position': boundary['position'],
                'strength': boundary['strength'],
                'type': boundary.get('type', 'attention_drop')
            })
        
        # Extract and store attention patterns
        pattern = self._extract_attention_pattern(window_data['layers'])
        self.attention_patterns.append({
            'window_index': window_idx,
            'pattern': pattern,
            'is_compressed': window_data.get('low_rank', False)
        })
        
    def _detect_boundaries_in_window(self, layers: List[Dict], tokens: List[str], 
                                   token_offset: int) -> List[Dict]:
        """
        Detect segment boundaries within a window using attention patterns.
        
        Args:
            layers: List of layer attention data
            tokens: List of tokens in window
            token_offset: Starting token index in full document
            
        Returns:
            List of detected boundaries
        """
        if not layers or not tokens:
            return []
            
        boundaries = []
        
        # Aggregate attention across layers (weighted by layer depth)
        aggregated_attention = None
        total_weight = 0
        
        for layer_data in layers:
            layer_idx = layer_data['layer_idx']
            attention = layer_data['attention']
            
            # Weight higher layers more (they contain semantic information)
            weight = 1.0 + (layer_idx / len(layers))
            
            if aggregated_attention is None:
                aggregated_attention = attention * weight
            else:
                aggregated_attention += attention * weight
            total_weight += weight
            
        aggregated_attention /= total_weight
        
        # Compute attention flow (how much each token attends to previous tokens)
        attention_flow = self._compute_attention_flow(aggregated_attention)
        
        # Find discontinuities in attention flow
        for i in range(1, len(attention_flow) - 1):
            # Look for drops in attention to previous context
            if i > 2:
                prev_flow = np.mean(attention_flow[i-3:i])
                curr_flow = attention_flow[i]
                
                if prev_flow > 0 and curr_flow / prev_flow < (1 - self.boundary_threshold):
                    # Significant drop in attention flow
                    boundaries.append({
                        'position': token_offset + i,
                        'strength': 1.0 - (curr_flow / prev_flow),
                        'type': 'attention_drop'
                    })
            
            # Look for attention peaks (new topic focus)
            if i < len(attention_flow) - 3:
                local_mean = np.mean(attention_flow[max(0, i-2):min(len(attention_flow), i+3)])
                if attention_flow[i] > local_mean * (1 + self.boundary_threshold):
                    boundaries.append({
                        'position': token_offset + i,
                        'strength': (attention_flow[i] / local_mean) - 1.0,
                        'type': 'attention_peak'
                    })
        
        return boundaries
    
    def _compute_attention_flow(self, attention_matrix: np.ndarray) -> np.ndarray:
        """
        Compute attention flow - how much each token attends to previous tokens.
        
        Args:
            attention_matrix: Shape [num_heads, seq_len, seq_len]
            
        Returns:
            Array of attention flow values per token
        """
        # Average across heads
        avg_attention = np.mean(attention_matrix, axis=0)
        
        # Compute backward attention flow
        # For each token, sum attention to all previous tokens
        seq_len = avg_attention.shape[0]
        attention_flow = np.zeros(seq_len)
        
        for i in range(seq_len):
            if i > 0:
                # Sum of attention from token i to all previous tokens
                attention_flow[i] = np.sum(avg_attention[i, :i])
        
        return attention_flow
    
    def _extract_attention_pattern(self, layers: List[Dict]) -> Dict[str, Any]:
        """
        Extract a compact representation of attention patterns.
        
        Args:
            layers: List of layer attention data
            
        Returns:
            Dictionary with attention pattern features
        """
        patterns = {
            'layer_patterns': [],
            'cross_layer_coherence': 0.0,
            'attention_entropy': []
        }
        
        for layer_data in layers:
            attention = layer_data['attention']
            
            # Compute attention entropy (measure of focus vs. dispersion)
            entropy = self._compute_attention_entropy(attention)
            patterns['attention_entropy'].append(entropy)
            
            # Extract dominant attention paths
            dominant_paths = self._extract_dominant_paths(attention)
            patterns['layer_patterns'].append({
                'layer_idx': layer_data['layer_idx'],
                'entropy': entropy,
                'dominant_paths': dominant_paths
            })
        
        # Compute cross-layer coherence
        if len(patterns['layer_patterns']) > 1:
            patterns['cross_layer_coherence'] = self._compute_cross_layer_coherence(
                patterns['layer_patterns']
            )
        
        return patterns
    
    def _compute_attention_entropy(self, attention_matrix: np.ndarray) -> float:
        """
        Compute entropy of attention distribution.
        
        Args:
            attention_matrix: Shape [num_heads, seq_len, seq_len]
            
        Returns:
            Average entropy across heads
        """
        eps = 1e-10
        entropies = []
        
        for head_idx in range(attention_matrix.shape[0]):
            # Normalize each row to probability distribution
            head_attention = attention_matrix[head_idx]
            
            # Compute entropy for each token's attention distribution
            token_entropies = []
            for i in range(head_attention.shape[0]):
                probs = head_attention[i] + eps
                probs = probs / probs.sum()
                entropy = -np.sum(probs * np.log(probs))
                token_entropies.append(entropy)
            
            entropies.append(np.mean(token_entropies))
        
        return np.mean(entropies)
    
    def _extract_dominant_paths(self, attention_matrix: np.ndarray, top_k: int = 3) -> List[Tuple[int, int, float]]:
        """
        Extract the most dominant attention paths.
        
        Args:
            attention_matrix: Shape [num_heads, seq_len, seq_len]
            top_k: Number of top paths to extract
            
        Returns:
            List of (from_token, to_token, weight) tuples
        """
        # Average across heads
        avg_attention = np.mean(attention_matrix, axis=0)
        
        # Find top-k attention values
        flat_indices = np.argpartition(avg_attention.ravel(), -top_k)[-top_k:]
        top_indices = np.unravel_index(flat_indices, avg_attention.shape)
        
        paths = []
        for i in range(top_k):
            from_token = top_indices[0][i]
            to_token = top_indices[1][i]
            weight = avg_attention[from_token, to_token]
            paths.append((int(from_token), int(to_token), float(weight)))
        
        return sorted(paths, key=lambda x: x[2], reverse=True)
    
    def _compute_cross_layer_coherence(self, layer_patterns: List[Dict]) -> float:
        """
        Compute coherence of attention patterns across layers.
        
        Args:
            layer_patterns: List of pattern dictionaries per layer
            
        Returns:
            Coherence score between 0 and 1
        """
        if len(layer_patterns) < 2:
            return 1.0
            
        # Compare entropy patterns across layers
        entropies = [p['entropy'] for p in layer_patterns]
        
        # Coherence is inverse of variance in entropy
        entropy_variance = np.var(entropies)
        coherence = 1.0 / (1.0 + entropy_variance)
        
        return float(coherence)
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get the accumulated results from all processed windows.
        
        Returns:
            Dictionary containing:
                - segment_boundaries: List of detected boundaries
                - attention_summary: Summary of attention patterns
                - suggested_segments: Recommended segmentation
        """
        # Consolidate boundaries across windows
        consolidated_boundaries = self._consolidate_boundaries()
        
        # Generate suggested segments
        suggested_segments = self._generate_segments(consolidated_boundaries)
        
        # Create attention summary
        attention_summary = {
            'total_windows': len(self.attention_patterns),
            'boundary_count': len(consolidated_boundaries),
            'average_entropy': self._compute_average_entropy(),
            'pattern_coherence': self._compute_overall_coherence()
        }
        
        return {
            'segment_boundaries': consolidated_boundaries,
            'attention_summary': attention_summary,
            'suggested_segments': suggested_segments
        }
    
    def _consolidate_boundaries(self) -> List[Dict]:
        """
        Consolidate boundaries from overlapping windows.
        
        Returns:
            List of consolidated boundary positions
        """
        if not self.window_boundaries:
            return []
            
        # Group boundaries by position
        position_groups = {}
        for boundary in self.window_boundaries:
            pos = boundary['position']
            if pos not in position_groups:
                position_groups[pos] = []
            position_groups[pos].append(boundary)
        
        # Consolidate overlapping boundaries
        consolidated = []
        for pos, boundaries in position_groups.items():
            # Average strength across detections
            avg_strength = np.mean([b['strength'] for b in boundaries])
            
            # Determine boundary type (prefer attention_drop over attention_peak)
            types = [b['type'] for b in boundaries]
            boundary_type = 'attention_drop' if 'attention_drop' in types else types[0]
            
            consolidated.append({
                'position': pos,
                'strength': float(avg_strength),
                'type': boundary_type,
                'detection_count': len(boundaries)
            })
        
        # Sort by position
        consolidated.sort(key=lambda x: x['position'])
        
        # Filter weak boundaries
        strength_threshold = 0.2
        consolidated = [b for b in consolidated if b['strength'] > strength_threshold]
        
        return consolidated
    
    def _generate_segments(self, boundaries: List[Dict]) -> List[Dict]:
        """
        Generate suggested segments based on boundaries.
        
        Args:
            boundaries: List of boundary positions
            
        Returns:
            List of suggested segment ranges
        """
        if not boundaries:
            return [{'start': 0, 'end': -1, 'confidence': 1.0}]
            
        segments = []
        start = 0
        
        for boundary in boundaries:
            end = boundary['position']
            
            # Only create segment if it's long enough
            if end - start >= self.min_segment_length:
                segments.append({
                    'start': start,
                    'end': end,
                    'confidence': boundary['strength']
                })
                start = end
        
        # Add final segment
        segments.append({
            'start': start,
            'end': -1,  # -1 indicates end of document
            'confidence': 0.8  # Default confidence for final segment
        })
        
        return segments
    
    def _compute_average_entropy(self) -> float:
        """Compute average entropy across all processed windows."""
        if not self.attention_patterns:
            return 0.0
            
        all_entropies = []
        for pattern in self.attention_patterns:
            if 'pattern' in pattern and 'attention_entropy' in pattern['pattern']:
                all_entropies.extend(pattern['pattern']['attention_entropy'])
        
        return float(np.mean(all_entropies)) if all_entropies else 0.0
    
    def _compute_overall_coherence(self) -> float:
        """Compute overall pattern coherence across windows."""
        if not self.attention_patterns:
            return 1.0
            
        coherences = []
        for pattern in self.attention_patterns:
            if 'pattern' in pattern and 'cross_layer_coherence' in pattern['pattern']:
                coherences.append(pattern['pattern']['cross_layer_coherence'])
        
        return float(np.mean(coherences)) if coherences else 1.0

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

    def process_dual_level_window(self, window_data: Dict[str, Any]) -> None:
        """
        Process a window with dual-level attention data.
        
        Args:
            window_data: Window data containing both token and node attention
        """
        # First process token-level (using base class method)
        self.process_window(window_data)
        
        if not window_data.get('dual_level', False):
            return
        
        # Extract dual-level specific data
        metadata = window_data.get('metadata', {})
        window_idx = metadata.get('window_index', 0)
        
        # Process each layer's dual-level patterns
        for layer_data in window_data.get('layers', []):
            if not self._is_dual_level_layer(layer_data):
                continue
            
            # Analyze token-level attention
            token_attention = layer_data.get('token_attention')
            if token_attention is not None:
                token_patterns = self._analyze_token_patterns(token_attention)
            
            # Analyze node-level attention
            node_attention = layer_data.get('node_attention')
            if node_attention is not None:
                node_patterns = self._analyze_node_patterns(node_attention)
                self.node_patterns.append({
                    'window_index': window_idx,
                    'layer_idx': layer_data['layer_idx'],
                    'patterns': node_patterns
                })
            
            # Analyze cross-level patterns
            cross_attention = layer_data.get('cross_attention')
            if cross_attention is not None:
                cross_patterns = self._analyze_cross_level_patterns(
                    cross_attention,
                    window_data.get('token_to_segment')
                )
                self.cross_level_patterns.append({
                    'window_index': window_idx,
                    'layer_idx': layer_data['layer_idx'],
                    'patterns': cross_patterns
                })
            
            # Compute segment cohesion
            if node_attention is not None and token_attention is not None:
                cohesion = self._compute_segment_cohesion(
                    token_attention,
                    node_attention,
                    window_data.get('token_to_segment')
                )
                self.segment_cohesion_scores.append({
                    'window_index': window_idx,
                    'layer_idx': layer_data['layer_idx'],
                    'cohesion': cohesion
                })
        
        # Detect dual-level boundaries
        dual_boundaries = self._detect_dual_level_boundaries(window_data)
        self.dual_level_boundaries.extend(dual_boundaries)
    
    def _is_dual_level_layer(self, layer_data: Dict[str, Any]) -> bool:
        """Check if layer contains dual-level data."""
        return layer_data.get('is_dual_level', False) or (
            'node_attention' in layer_data or 'cross_attention' in layer_data
        )
    
    def _analyze_token_patterns(self, attention: np.ndarray) -> Dict[str, Any]:
        """Analyze token-level attention patterns."""
        # Use base class methods
        entropy = self._compute_attention_entropy(torch.from_numpy(attention))
        flow = self._compute_attention_flow(attention.mean(axis=0) if len(attention.shape) > 2 else attention)
        
        return {
            'entropy': float(entropy),
            'mean_flow': float(flow.mean()),
            'flow_variance': float(flow.var())
        }
    
    def _analyze_node_patterns(self, node_attention: np.ndarray) -> Dict[str, Any]:
        """
        Analyze node-level attention patterns.
        
        Args:
            node_attention: Node attention matrix [num_heads, num_nodes, num_nodes]
                           or [num_nodes, num_nodes]
            
        Returns:
            Dictionary of node-level patterns
        """
        # Handle different shapes
        if len(node_attention.shape) == 3:
            # Average across heads
            node_attention = node_attention.mean(axis=0)
        
        num_nodes = node_attention.shape[0]
        
        # Compute node centrality
        in_degree = node_attention.sum(axis=0)
        out_degree = node_attention.sum(axis=1)
        
        # Find hub nodes (high in/out degree)
        hub_threshold = node_attention.mean() + node_attention.std()
        hub_nodes = np.where((in_degree > hub_threshold) | (out_degree > hub_threshold))[0]
        
        # Compute clustering coefficient
        clustering_coeff = self._compute_clustering_coefficient(node_attention)
        
        # Find strongly connected components
        strong_connections = (node_attention > hub_threshold).astype(int)
        components = self._find_connected_components(strong_connections)
        
        return {
            'num_nodes': num_nodes,
            'mean_attention': float(node_attention.mean()),
            'attention_variance': float(node_attention.var()),
            'hub_nodes': hub_nodes.tolist(),
            'num_hubs': len(hub_nodes),
            'clustering_coefficient': float(clustering_coeff),
            'num_components': len(components),
            'largest_component_size': max(len(c) for c in components) if components else 0,
            'graph_density': float((node_attention > 0).sum() / (num_nodes * num_nodes))
        }
    
    def _analyze_cross_level_patterns(self, cross_attention: np.ndarray,
                                    token_to_segment: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze cross-level attention patterns (tokens attending to nodes).
        
        Args:
            cross_attention: Cross attention matrix [num_heads, num_tokens, num_nodes]
                           or [num_tokens, num_nodes]
            token_to_segment: Token to segment mapping
            
        Returns:
            Dictionary of cross-level patterns
        """
        # Handle different shapes
        if len(cross_attention.shape) == 3:
            cross_attention = cross_attention.mean(axis=0)
        
        num_tokens, num_nodes = cross_attention.shape
        
        # Analyze attention distribution
        # Which nodes do tokens attend to most?
        dominant_nodes = cross_attention.argmax(axis=1)
        attention_strengths = cross_attention.max(axis=1)
        
        # Compute attention consistency within segments
        consistency = 0.0
        if token_to_segment is not None:
            for node_id in range(num_nodes):
                node_tokens = np.where(token_to_segment == node_id)[0]
                if len(node_tokens) > 1:
                    # How consistently do tokens in this segment attend to their node?
                    node_attention = cross_attention[node_tokens, node_id]
                    consistency += node_attention.mean()
            consistency /= num_nodes
        
        return {
            'mean_cross_attention': float(cross_attention.mean()),
            'max_cross_attention': float(cross_attention.max()),
            'attention_consistency': float(consistency),
            'dominant_node_distribution': np.bincount(dominant_nodes, minlength=num_nodes).tolist(),
            'mean_attention_strength': float(attention_strengths.mean())
        }
    
    def _compute_segment_cohesion(self, token_attention: np.ndarray,
                                node_attention: np.ndarray,
                                token_to_segment: Optional[np.ndarray]) -> Dict[str, float]:
        """
        Compute how cohesive segments are based on dual-level attention.
        
        Args:
            token_attention: Token-level attention
            node_attention: Node-level attention
            token_to_segment: Token to segment mapping
            
        Returns:
            Dictionary of cohesion scores
        """
        if token_to_segment is None:
            return {'overall_cohesion': 0.0}
        
        # Average across heads if needed
        if len(token_attention.shape) > 2:
            token_attention = token_attention.mean(axis=0)
        if len(node_attention.shape) > 2:
            node_attention = node_attention.mean(axis=0)
        
        num_segments = int(token_to_segment.max()) + 1
        segment_cohesions = []
        
        for seg_id in range(num_segments):
            seg_tokens = np.where(token_to_segment == seg_id)[0]
            if len(seg_tokens) < 2:
                continue
            
            # Internal token attention within segment
            internal_attention = token_attention[np.ix_(seg_tokens, seg_tokens)]
            internal_cohesion = internal_attention.mean()
            
            # External token attention (to other segments)
            external_mask = ~np.isin(np.arange(len(token_to_segment)), seg_tokens)
            if external_mask.any():
                external_attention = token_attention[np.ix_(seg_tokens, external_mask)]
                external_cohesion = external_attention.mean()
                
                # Cohesion ratio: internal vs external
                cohesion_ratio = internal_cohesion / (external_cohesion + 1e-6)
            else:
                cohesion_ratio = internal_cohesion
            
            segment_cohesions.append(cohesion_ratio)
        
        return {
            'overall_cohesion': float(np.mean(segment_cohesions)) if segment_cohesions else 0.0,
            'cohesion_variance': float(np.var(segment_cohesions)) if segment_cohesions else 0.0,
            'min_cohesion': float(np.min(segment_cohesions)) if segment_cohesions else 0.0,
            'max_cohesion': float(np.max(segment_cohesions)) if segment_cohesions else 0.0
        }
    
    def _detect_dual_level_boundaries(self, window_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect boundaries using both token and node-level patterns.
        
        Args:
            window_data: Window data with dual-level attention
            
        Returns:
            List of detected boundaries with confidence scores
        """
        boundaries = []
        metadata = window_data.get('metadata', {})
        token_offset = metadata.get('token_start_index', 0)
        
        # Get token-to-segment mapping
        token_to_segment = window_data.get('token_to_segment')
        if token_to_segment is None:
            return boundaries
        
        # Find segment transitions
        for i in range(1, len(token_to_segment)):
            if token_to_segment[i] != token_to_segment[i-1]:
                # Segment boundary detected
                boundaries.append({
                    'position': token_offset + i,
                    'strength': 1.0,  # High confidence for explicit boundaries
                    'type': 'segment_transition',
                    'from_segment': int(token_to_segment[i-1]),
                    'to_segment': int(token_to_segment[i])
                })
        
        # Also detect boundaries from attention patterns
        for layer_data in window_data.get('layers', []):
            if not layer_data.get('node_attention') is not None:
                continue
            
            # Look for weak connections between nodes
            node_attention = layer_data['node_attention']
            if len(node_attention.shape) > 2:
                node_attention = node_attention.mean(axis=0)
            
            weak_threshold = node_attention.mean() - node_attention.std()
            
            for i in range(node_attention.shape[0]):
                for j in range(i+1, node_attention.shape[1]):
                    if node_attention[i, j] < weak_threshold and node_attention[j, i] < weak_threshold:
                        # Weak connection between segments
                        # Find boundary position
                        seg_i_tokens = np.where(token_to_segment == i)[0]
                        seg_j_tokens = np.where(token_to_segment == j)[0]
                        
                        if len(seg_i_tokens) > 0 and len(seg_j_tokens) > 0:
                            boundary_pos = (seg_i_tokens[-1] + seg_j_tokens[0]) // 2
                            boundaries.append({
                                'position': token_offset + boundary_pos,
                                'strength': 1.0 - (node_attention[i, j] + node_attention[j, i]) / 2,
                                'type': 'weak_connection',
                                'segments': [i, j]
                            })
        
        return boundaries
    
    def _compute_clustering_coefficient(self, adjacency: np.ndarray) -> float:
        """
        Compute clustering coefficient for directed graph.
        
        Args:
            adjacency: Adjacency matrix
            
        Returns:
            Average clustering coefficient
        """
        n = adjacency.shape[0]
        if n < 3:
            return 0.0
        
        clustering_coeffs = []
        
        for i in range(n):
            # Find neighbors
            neighbors = np.where(adjacency[i] > 0)[0]
            k = len(neighbors)
            
            if k < 2:
                continue
            
            # Count edges between neighbors
            edge_count = 0
            for j in neighbors:
                for l in neighbors:
                    if j != l and adjacency[j, l] > 0:
                        edge_count += 1
            
            # Clustering coefficient for node i
            max_edges = k * (k - 1)
            if max_edges > 0:
                clustering_coeffs.append(edge_count / max_edges)
        
        return np.mean(clustering_coeffs) if clustering_coeffs else 0.0
    
    def _find_connected_components(self, adjacency: np.ndarray) -> List[List[int]]:
        """
        Find strongly connected components in directed graph.
        
        Args:
            adjacency: Binary adjacency matrix
            
        Returns:
            List of components (each component is a list of node indices)
        """
        n = adjacency.shape[0]
        visited = np.zeros(n, dtype=bool)
        components = []
        
        def dfs(node, component):
            visited[node] = True
            component.append(node)
            
            # Check outgoing and incoming edges
            for neighbor in range(n):
                if not visited[neighbor]:
                    if adjacency[node, neighbor] > 0 or adjacency[neighbor, node] > 0:
                        dfs(neighbor, component)
        
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i, component)
                components.append(component)
        
        return components
    
    def get_dual_level_results(self) -> Dict[str, Any]:
        """
        Get comprehensive results including dual-level analysis.
        
        Returns:
            Dictionary with both token and node-level results
        """
        # Get base results
        base_results = self.get_results()
        
        # Add dual-level specific results
        dual_results = {
            **base_results,
            'dual_level_analysis': {
                'node_patterns': self._aggregate_node_patterns(),
                'cross_level_patterns': self._aggregate_cross_patterns(),
                'segment_cohesion': self._aggregate_cohesion_scores(),
                'dual_boundaries': self._consolidate_dual_boundaries(),
                'graph_metrics': self._compute_graph_metrics()
            }
        }
        
        return dual_results
    
    def _aggregate_node_patterns(self) -> Dict[str, Any]:
        """Aggregate node-level patterns across windows."""
        if not self.node_patterns:
            return {}
        
        all_patterns = [p['patterns'] for p in self.node_patterns]
        
        return {
            'mean_node_attention': np.mean([p['mean_attention'] for p in all_patterns]),
            'mean_clustering': np.mean([p['clustering_coefficient'] for p in all_patterns]),
            'mean_density': np.mean([p['graph_density'] for p in all_patterns]),
            'total_hubs': sum(p['num_hubs'] for p in all_patterns),
            'avg_component_size': np.mean([p['largest_component_size'] for p in all_patterns])
        }
    
    def _aggregate_cross_patterns(self) -> Dict[str, Any]:
        """Aggregate cross-level patterns."""
        if not self.cross_level_patterns:
            return {}
        
        all_patterns = [p['patterns'] for p in self.cross_level_patterns]
        
        return {
            'mean_cross_attention': np.mean([p['mean_cross_attention'] for p in all_patterns]),
            'mean_consistency': np.mean([p['attention_consistency'] for p in all_patterns]),
            'mean_strength': np.mean([p['mean_attention_strength'] for p in all_patterns])
        }
    
    def _aggregate_cohesion_scores(self) -> Dict[str, Any]:
        """Aggregate segment cohesion scores."""
        if not self.segment_cohesion_scores:
            return {}
        
        all_cohesions = [s['cohesion']['overall_cohesion'] for s in self.segment_cohesion_scores]
        
        return {
            'mean_cohesion': float(np.mean(all_cohesions)),
            'cohesion_variance': float(np.var(all_cohesions)),
            'min_cohesion': float(np.min(all_cohesions)),
            'max_cohesion': float(np.max(all_cohesions))
        }
    
    def _consolidate_dual_boundaries(self) -> List[Dict[str, Any]]:
        """Consolidate dual-level boundaries with base boundaries."""
        # Combine with base boundaries
        all_boundaries = self.window_boundaries + self.dual_level_boundaries
        
        # Sort by position
        all_boundaries.sort(key=lambda x: x['position'])
        
        # Merge nearby boundaries
        consolidated = []
        merge_distance = 5  # tokens
        
        i = 0
        while i < len(all_boundaries):
            current = all_boundaries[i]
            
            # Find all boundaries within merge distance
            j = i + 1
            while j < len(all_boundaries) and all_boundaries[j]['position'] - current['position'] <= merge_distance:
                j += 1
            
            if j > i + 1:
                # Merge boundaries
                merged_strength = max(b['strength'] for b in all_boundaries[i:j])
                merged_types = list(set(b['type'] for b in all_boundaries[i:j]))
                
                consolidated.append({
                    'position': current['position'],
                    'strength': merged_strength,
                    'type': '+'.join(merged_types),
                    'merged_count': j - i
                })
            else:
                consolidated.append(current)
            
            i = j
        
        return consolidated
    
    def _compute_graph_metrics(self) -> Dict[str, Any]:
        """Compute overall graph metrics from dual-level processing."""
        if not self.node_patterns:
            return {}
        
        return {
            'windows_processed': len(self.node_patterns),
            'has_node_attention': len(self.node_patterns) > 0,
            'has_cross_attention': len(self.cross_level_patterns) > 0,
            'dual_boundary_count': len(self.dual_level_boundaries)
        }
