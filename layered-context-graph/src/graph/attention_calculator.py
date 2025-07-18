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

logger = logging.getLogger(__name__)


class AttentionCalculator:
    """
    A stateful calculator for processing attention windows with low-rank support.
    Compatible with the windowing approach used by QwQModel and PartitionManager.
    """
    
    def __init__(self, rank_ratio: float = 0.1, top_n_layers: int = 4, 
                 boundary_threshold: float = 0.3, min_segment_length: int = 50):
        """
        Initialize the attention calculator.
        
        Args:
            rank_ratio: Compression ratio for low-rank decomposition
            top_n_layers: Number of top layers to process
            boundary_threshold: Threshold for detecting segment boundaries
            min_segment_length: Minimum characters per segment
        """
        self.rank_ratio = rank_ratio
        self.top_n_layers = top_n_layers
        self.boundary_threshold = boundary_threshold
        self.min_segment_length = min_segment_length
        
        # State for accumulating results
        self.window_boundaries = []
        self.attention_patterns = []
        self.token_positions = []
        self.aggregate_attention_flow = None
        
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