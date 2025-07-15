#!/usr/bin/env python3
"""
Consolidated Edge Detector
==========================
Detects various types of relationships between nodes to build rich graphs,
combining rule-based and attention-based methods.
"""

import re
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class EdgeDetector:
    """
    Detects multiple types of edges between nodes using both rule-based
    and attention-based approaches.
    """
    
    def __init__(self, attention_extractor=None, attention_threshold: float = 0.15):
        """
        Initialize the consolidated edge detector.
        
        Args:
            attention_extractor: Instance for extracting attention patterns.
            attention_threshold: Minimum attention score to create an edge.
        """
        self.attention_extractor = attention_extractor
        self.attention_threshold = attention_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def detect_edges(self, nodes: List[Dict[str, Any]], use_attention: bool = True) -> List[Dict[str, Any]]:
        """
        Detect all types of edges between nodes.
        If use_attention is True and an attention_extractor is available,
        it will prioritize attention-based edge detection.
        """
        if use_attention and self.attention_extractor:
            logger.info("Using attention-based edge detection.")
            # Extract contents for attention processing
            contents = [node.get('content', '') for node in nodes]
            attention_patterns = self.attention_extractor.extract_attention_for_tape_splitting(contents)
            
            if attention_patterns:
                return self._detect_edges_from_attention(nodes, attention_patterns)
            else:
                logger.warning("Attention extraction failed, falling back to rule-based detection.")
                return self._detect_edges_rule_based(nodes)
        else:
            logger.info("Using rule-based edge detection.")
            return self._detect_edges_rule_based(nodes)

    # --- Attention-Based Methods ---

    def _detect_edges_from_attention(self, 
                                   nodes: List[Dict[str, Any]], 
                                   attention_patterns: Dict) -> List[Dict[str, Any]]:
        """
        Detect edges using pre-computed attention patterns.
        """
        edges = []
        
        # Process different types of attention patterns
        if 'window_patterns' in attention_patterns:
            edges.extend(self._edges_from_window_patterns(nodes, attention_patterns['window_patterns']))
        
        if 'qwq_attention_patterns' in attention_patterns:
            edges.extend(self._edges_from_qwq_attention(nodes, attention_patterns['qwq_attention_patterns']))
        
        # Fallback for basic connectivity
        if not edges:
            logger.warning("No edges detected from attention, creating sequential fallback.")
            edges.extend(self._create_sequential_edges(nodes))

        return self._deduplicate_edges(edges)

    def _edges_from_window_patterns(self, nodes: List[Dict[str, Any]], window_patterns: List[Dict]) -> List[Dict[str, Any]]:
        """Extract edges from windowed attention patterns."""
        edges = []
        for window in window_patterns:
            attention_matrix = window.get('attention_matrix')
            if attention_matrix is None:
                continue
            
            attention_matrix = torch.tensor(attention_matrix, device=self.device)
            if attention_matrix.dim() > 2:
                attention_matrix = attention_matrix.mean(dim=0)

            for i in range(attention_matrix.shape[0]):
                for j in range(attention_matrix.shape[1]):
                    if i >= j: continue
                    
                    score = attention_matrix[i, j].item()
                    if score > self.attention_threshold:
                        edges.append({
                            'source': nodes[i]['id'],
                            'target': nodes[j]['id'],
                            'weight': float(score),
                            'type': 'attention_based',
                            'attention_based': True
                        })
        return edges

    def _edges_from_qwq_attention(self, nodes: List[Dict[str, Any]], qwq_patterns: Dict) -> List[Dict[str, Any]]:
        """Extract edges from QwQ-specific attention patterns."""
        # This method can be expanded with the detailed logic from the original file if needed.
        return []

    # --- Rule-Based Methods ---

    def _detect_edges_rule_based(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detects various types of edges using heuristic rules.
        """
        edges = []
        edge_detectors = [
            self._detect_sequential, self._detect_reference, self._detect_elaboration,
            self._detect_contrast, self._detect_causal, self._detect_example,
            self._detect_question_answer, self._detect_summary, self._detect_code_explanation,
            self._detect_semantic_similarity
        ]

        for i, source_node in enumerate(nodes):
            for j, target_node in enumerate(nodes):
                if i >= j: continue
                for detector in edge_detectors:
                    edge = detector(source_node, target_node, i, j)
                    if edge:
                        edges.append(edge)
        
        return self._deduplicate_edges(edges)

    def _detect_sequential(self, source: Dict, target: Dict, source_idx: int, target_idx: int) -> Optional[Dict]:
        if abs(source_idx - target_idx) == 1:
            return {'source': source['id'], 'target': target['id'], 'weight': 0.8, 'type': 'sequential'}
        return None

    def _detect_reference(self, source: Dict, target: Dict, source_idx: int, target_idx: int) -> Optional[Dict]:
        content = target.get('content', '').lower()
        if any(p in content for p in ['as mentioned', 'see above', 'earlier']):
            return {'source': source['id'], 'target': target['id'], 'weight': 0.7, 'type': 'reference'}
        return None

    def _detect_elaboration(self, source: Dict, target: Dict, source_idx: int, target_idx: int) -> Optional[Dict]:
        content = target.get('content', '').lower()
        if any(m in content for m in ['furthermore', 'additionally', 'in addition']) and target_idx == source_idx + 1:
            return {'source': source['id'], 'target': target['id'], 'weight': 0.8, 'type': 'elaboration'}
        return None

    def _detect_contrast(self, source: Dict, target: Dict, source_idx: int, target_idx: int) -> Optional[Dict]:
        content = target.get('content', '').lower()
        if any(m in content for m in ['however', 'but', 'in contrast']):
            return {'source': source['id'], 'target': target['id'], 'weight': 0.7, 'type': 'contrast'}
        return None

    def _detect_causal(self, source: Dict, target: Dict, source_idx: int, target_idx: int) -> Optional[Dict]:
        content = target.get('content', '').lower()
        if any(m in content for m in ['therefore', 'as a result', 'because']):
            return {'source': source['id'], 'target': target['id'], 'weight': 0.9, 'type': 'causal'}
        return None

    def _detect_example(self, source: Dict, target: Dict, source_idx: int, target_idx: int) -> Optional[Dict]:
        content = target.get('content', '').lower()
        if any(m in content for m in ['for example', 'for instance']):
            return {'source': source['id'], 'target': target['id'], 'weight': 0.8, 'type': 'example'}
        return None

    def _detect_question_answer(self, source: Dict, target: Dict, source_idx: int, target_idx: int) -> Optional[Dict]:
        if source.get('content', '').strip().endswith('?'):
            return {'source': source['id'], 'target': target['id'], 'weight': 0.9, 'type': 'question_answer'}
        return None

    def _detect_summary(self, source: Dict, target: Dict, source_idx: int, target_idx: int) -> Optional[Dict]:
        content = target.get('content', '').lower()
        if any(m in content for m in ['in summary', 'to summarize', 'in conclusion']):
            return {'source': source['id'], 'target': target['id'], 'weight': 0.7, 'type': 'summary'}
        return None

    def _detect_code_explanation(self, source: Dict, target: Dict, source_idx: int, target_idx: int) -> Optional[Dict]:
        source_has_code = '```' in source.get('content', '')
        target_has_code = '```' in target.get('content', '')
        if source_has_code and not target_has_code and target_idx == source_idx + 1:
            return {'source': source['id'], 'target': target['id'], 'weight': 0.8, 'type': 'code_explanation'}
        if not source_has_code and target_has_code and target_idx == source_idx + 1:
            return {'source': source['id'], 'target': target['id'], 'weight': 0.8, 'type': 'explanation_code'}
        return None

    def _detect_semantic_similarity(self, source: Dict, target: Dict, source_idx: int, target_idx: int) -> Optional[Dict]:
        # A simple word overlap for demonstration. A real implementation would use embeddings.
        source_words = set(source.get('content', '').lower().split())
        target_words = set(target.get('content', '').lower().split())
        common_words = source_words.intersection(target_words)
        similarity = len(common_words) / (len(source_words) + len(target_words) - len(common_words) + 1e-6)
        if similarity > 0.4:
            return {'source': source['id'], 'target': target['id'], 'weight': similarity, 'type': 'semantic_similarity'}
        return None

    # --- Helper Methods ---

    def _create_sequential_edges(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create basic sequential edges as a fallback."""
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
        """Remove duplicate edges, keeping the one with the highest weight."""
        edge_map = {}
        for edge in edges:
            key = tuple(sorted((edge['source'], edge['target'])))
            if key not in edge_map or edge['weight'] > edge_map[key]['weight']:
                edge_map[key] = edge
        return list(edge_map.values())
