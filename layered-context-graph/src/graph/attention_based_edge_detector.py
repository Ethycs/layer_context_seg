#!/usr/bin/env python3
"""
Attention-Based Edge Detector
=============================
Uses transformer attention patterns from QwQ to detect relationships between nodes.
This replaces pure algorithmic edge detection with LLM-derived connections.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AttentionBasedEdgeDetector:
    """
    Detects edges between nodes using transformer attention patterns.
    Leverages QwQ model's understanding of relationships rather than algorithmic rules.
    """
    
    def __init__(self, attention_extractor=None, threshold: float = 0.15):
        """
        Initialize attention-based edge detector.
        
        Args:
            attention_extractor: EnhancedAttentionExtractor instance with QwQ model
            threshold: Minimum attention score to create an edge
        """
        self.attention_extractor = attention_extractor
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def detect_edges_from_attention(self, 
                                   nodes: List[Dict[str, Any]], 
                                   attention_patterns: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Detect edges using attention patterns from the transformer model.
        
        Args:
            nodes: List of node dictionaries with content
            attention_patterns: Pre-computed attention patterns (optional)
            
        Returns:
            List of edges with attention-based weights and types
        """
        edges = []
        
        # Extract node contents
        contents = [node.get('content', '') for node in nodes]
        
        # Get attention patterns if not provided
        if attention_patterns is None and self.attention_extractor:
            logger.info("Extracting attention patterns for edge detection...")
            attention_patterns = self.attention_extractor.extract_attention_for_tape_splitting(contents)
        
        if not attention_patterns:
            logger.warning("No attention patterns available, falling back to sequential edges")
            return self._create_sequential_edges(nodes)
        
        # Process attention patterns to create edges
        if 'window_patterns' in attention_patterns:
            edges.extend(self._edges_from_window_patterns(nodes, attention_patterns['window_patterns']))
        
        if 'qwq_attention_patterns' in attention_patterns:
            edges.extend(self._edges_from_qwq_attention(nodes, attention_patterns['qwq_attention_patterns']))
        
        # Deduplicate edges
        edges = self._deduplicate_edges(edges)
        
        return edges
    
    def _edges_from_window_patterns(self, 
                                   nodes: List[Dict[str, Any]], 
                                   window_patterns: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extract edges from windowed attention patterns.
        """
        edges = []
        
        for window in window_patterns:
            if 'attention_matrix' not in window:
                continue
                
            attention_matrix = window['attention_matrix']
            window_start = window.get('start_idx', 0)
            window_end = window.get('end_idx', len(nodes))
            
            # Convert to tensor if needed
            if isinstance(attention_matrix, list):
                attention_matrix = torch.tensor(attention_matrix)
            elif isinstance(attention_matrix, np.ndarray):
                attention_matrix = torch.from_numpy(attention_matrix)
            
            # Move to device
            attention_matrix = attention_matrix.to(self.device)
            
            # Average across attention heads if needed
            if attention_matrix.dim() > 2:
                attention_matrix = attention_matrix.mean(dim=0)
            
            # Create edges from high attention scores
            for i in range(window_start, min(window_end, attention_matrix.shape[0])):
                for j in range(window_start, min(window_end, attention_matrix.shape[1])):
                    if i >= j:  # Skip self and backward connections
                        continue
                    
                    attention_score = attention_matrix[i - window_start, j - window_start].item()
                    
                    if attention_score > self.threshold:
                        edge = {
                            'source': nodes[i]['id'],
                            'target': nodes[j]['id'],
                            'weight': float(attention_score),
                            'type': self._classify_edge_type(attention_score, i, j),
                            'attention_based': True
                        }
                        edges.append(edge)
        
        return edges
    
    def _edges_from_qwq_attention(self, 
                                 nodes: List[Dict[str, Any]], 
                                 qwq_patterns: Dict) -> List[Dict[str, Any]]:
        """
        Extract edges specifically from QwQ attention patterns.
        """
        edges = []
        
        if 'attention_weights' in qwq_patterns:
            weights = qwq_patterns['attention_weights']
            
            # Process multi-layer attention
            for layer_idx, layer_attention in enumerate(weights):
                if isinstance(layer_attention, dict):
                    # Extract attention matrices from layer
                    for head_name, attention_matrix in layer_attention.items():
                        if 'attn' in head_name.lower():
                            edges.extend(self._process_attention_head(
                                nodes, attention_matrix, layer_idx, head_name
                            ))
        
        return edges
    
    def _process_attention_head(self, 
                               nodes: List[Dict[str, Any]], 
                               attention_matrix: Any,
                               layer_idx: int,
                               head_name: str) -> List[Dict[str, Any]]:
        """
        Process a single attention head to extract edges.
        """
        edges = []
        
        # Convert to tensor
        if isinstance(attention_matrix, (list, np.ndarray)):
            attention_matrix = torch.tensor(attention_matrix, device=self.device)
        
        # Ensure 2D
        if attention_matrix.dim() > 2:
            attention_matrix = attention_matrix.squeeze()
        
        # Skip if wrong shape
        if attention_matrix.dim() != 2:
            return edges
        
        # Normalize attention scores
        attention_matrix = torch.softmax(attention_matrix, dim=-1)
        
        # Extract strong connections
        n = min(len(nodes), attention_matrix.shape[0])
        for i in range(n):
            for j in range(n):
                if i >= j:
                    continue
                
                score = attention_matrix[i, j].item()
                if score > self.threshold:
                    # Higher layer attention indicates more abstract relationships
                    edge_type = 'semantic' if layer_idx > 20 else 'syntactic'
                    
                    edges.append({
                        'source': nodes[i]['id'],
                        'target': nodes[j]['id'],
                        'weight': float(score),
                        'type': edge_type,
                        'layer': layer_idx,
                        'head': head_name,
                        'attention_based': True
                    })
        
        return edges
    
    def _classify_edge_type(self, attention_score: float, i: int, j: int) -> str:
        """
        Classify edge type based on attention pattern.
        """
        distance = j - i
        
        if distance == 1:
            return 'sequential'
        elif attention_score > 0.5:
            return 'strong_reference'
        elif attention_score > 0.3:
            return 'reference'
        elif distance > 5:
            return 'long_range'
        else:
            return 'local_context'
    
    def _create_sequential_edges(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create basic sequential edges as fallback.
        """
        edges = []
        for i in range(len(nodes) - 1):
            edges.append({
                'source': nodes[i]['id'],
                'target': nodes[i + 1]['id'],
                'weight': 0.5,
                'type': 'sequential',
                'attention_based': False
            })
        return edges
    
    def _deduplicate_edges(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate edges, keeping the one with highest weight.
        """
        edge_map = {}
        
        for edge in edges:
            key = (edge['source'], edge['target'])
            if key not in edge_map or edge['weight'] > edge_map[key]['weight']:
                edge_map[key] = edge
        
        return list(edge_map.values())
    
    def enhance_with_attention_metadata(self, 
                                       edges: List[Dict[str, Any]], 
                                       nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance existing edges with attention-based metadata.
        """
        if not self.attention_extractor:
            return edges
        
        # Get node contents
        contents = [node.get('content', '') for node in nodes]
        
        # Extract attention patterns
        attention_patterns = self.attention_extractor.extract_attention_for_tape_splitting(contents)
        
        # Create node ID to index mapping
        id_to_idx = {node['id']: i for i, node in enumerate(nodes)}
        
        # Enhance each edge
        for edge in edges:
            source_idx = id_to_idx.get(edge['source'])
            target_idx = id_to_idx.get(edge['target'])
            
            if source_idx is not None and target_idx is not None:
                # Look for attention score between these nodes
                attention_score = self._find_attention_score(
                    attention_patterns, source_idx, target_idx
                )
                
                if attention_score > 0:
                    edge['attention_score'] = float(attention_score)
                    edge['attention_enhanced'] = True
                    
                    # Adjust weight based on attention
                    if 'weight' in edge:
                        edge['weight'] = edge['weight'] * (1 + attention_score)
                    else:
                        edge['weight'] = attention_score
        
        return edges
    
    def _find_attention_score(self, 
                             attention_patterns: Dict, 
                             source_idx: int, 
                             target_idx: int) -> float:
        """
        Find attention score between two node indices.
        """
        if 'window_patterns' in attention_patterns:
            for window in attention_patterns['window_patterns']:
                if 'attention_matrix' in window:
                    matrix = window['attention_matrix']
                    if isinstance(matrix, (torch.Tensor, np.ndarray)):
                        if source_idx < len(matrix) and target_idx < len(matrix[0]):
                            return float(matrix[source_idx][target_idx])
        
        return 0.0