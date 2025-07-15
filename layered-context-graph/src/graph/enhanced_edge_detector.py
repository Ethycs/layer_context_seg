#!/usr/bin/env python3
"""
Enhanced Edge Detector - Comprehensive Relationship Detection
============================================================
Detects various types of relationships between nodes to build richer graphs
"""

import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import defaultdict


class EnhancedEdgeDetector:
    """
    Detects multiple types of edges between nodes
    """
    
    def __init__(self):
        self.edge_types = {
            'sequential': self._detect_sequential,
            'reference': self._detect_reference,
            'elaboration': self._detect_elaboration,
            'contrast': self._detect_contrast,
            'causal': self._detect_causal,
            'example': self._detect_example,
            'question_answer': self._detect_question_answer,
            'summary': self._detect_summary,
            'code_explanation': self._detect_code_explanation,
            'semantic_similarity': self._detect_semantic_similarity
        }
        
    def detect_edges(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect all types of edges between nodes
        """
        edges = []
        
        # Check each pair of nodes
        for i, source_node in enumerate(nodes):
            for j, target_node in enumerate(nodes):
                if i >= j:  # Skip self and already checked pairs
                    continue
                    
                # Check for each edge type
                for edge_type, detector in self.edge_types.items():
                    edge = detector(source_node, target_node, i, j)
                    if edge:
                        edge['type'] = edge_type
                        edges.append(edge)
        
        # Add sequential edges for basic connectivity
        edges.extend(self._add_sequential_edges(nodes))
        
        # Deduplicate edges
        edges = self._deduplicate_edges(edges)
        
        return edges
    
    def _detect_sequential(self, source: Dict, target: Dict, 
                          source_idx: int, target_idx: int) -> Optional[Dict]:
        """Detect sequential relationship"""
        # Adjacent nodes have sequential relationship
        if abs(source_idx - target_idx) == 1:
            return {
                'source': source['id'],
                'target': target['id'],
                'weight': 0.8,
                'metadata': {'distance': 1}
            }
        return None
    
    def _detect_reference(self, source: Dict, target: Dict, 
                         source_idx: int, target_idx: int) -> Optional[Dict]:
        """Detect reference relationships"""
        source_content = source.get('content', '').lower()
        target_content = target.get('content', '').lower()
        
        # Look for explicit references
        reference_patterns = [
            r'as mentioned',
            r'as discussed',
            r'see above',
            r'see below',
            r'earlier',
            r'later',
            r'previously',
            r'following'
        ]
        
        for pattern in reference_patterns:
            if re.search(pattern, source_content) or re.search(pattern, target_content):
                return {
                    'source': source['id'],
                    'target': target['id'],
                    'weight': 0.7,
                    'metadata': {'reference_type': 'explicit'}
                }
        
        return None
    
    def _detect_elaboration(self, source: Dict, target: Dict, 
                           source_idx: int, target_idx: int) -> Optional[Dict]:
        """Detect elaboration relationships"""
        source_content = source.get('content', '').lower()
        target_content = target.get('content', '').lower()
        
        elaboration_markers = [
            'furthermore',
            'additionally',
            'moreover',
            'in addition',
            'specifically',
            'in particular',
            'for example',
            'that is',
            'namely'
        ]
        
        for marker in elaboration_markers:
            if marker in target_content and target_idx == source_idx + 1:
                return {
                    'source': source['id'],
                    'target': target['id'],
                    'weight': 0.8,
                    'metadata': {'elaboration_marker': marker}
                }
        
        return None
    
    def _detect_contrast(self, source: Dict, target: Dict, 
                        source_idx: int, target_idx: int) -> Optional[Dict]:
        """Detect contrasting relationships"""
        source_content = source.get('content', '').lower()
        target_content = target.get('content', '').lower()
        
        contrast_markers = [
            'however',
            'but',
            'on the other hand',
            'in contrast',
            'conversely',
            'alternatively',
            'whereas',
            'while',
            'although'
        ]
        
        for marker in contrast_markers:
            if marker in target_content:
                return {
                    'source': source['id'],
                    'target': target['id'],
                    'weight': 0.7,
                    'metadata': {'contrast_marker': marker}
                }
        
        return None
    
    def _detect_causal(self, source: Dict, target: Dict, 
                      source_idx: int, target_idx: int) -> Optional[Dict]:
        """Detect causal relationships"""
        target_content = target.get('content', '').lower()
        
        causal_markers = [
            'therefore',
            'thus',
            'hence',
            'as a result',
            'consequently',
            'because',
            'since',
            'due to',
            'this leads to',
            'this causes'
        ]
        
        for marker in causal_markers:
            if marker in target_content:
                return {
                    'source': source['id'],
                    'target': target['id'],
                    'weight': 0.9,
                    'metadata': {'causal_marker': marker}
                }
        
        return None
    
    def _detect_example(self, source: Dict, target: Dict, 
                       source_idx: int, target_idx: int) -> Optional[Dict]:
        """Detect example relationships"""
        target_content = target.get('content', '').lower()
        
        example_markers = [
            'for example',
            'for instance',
            'such as',
            'like',
            'e.g.',
            'i.e.',
            'example:',
            'here\'s an example'
        ]
        
        # Check if target contains code block after source discusses concept
        has_code = '```' in target.get('content', '') or 'def ' in target.get('content', '')
        
        for marker in example_markers:
            if marker in target_content or (has_code and target_idx == source_idx + 1):
                return {
                    'source': source['id'],
                    'target': target['id'],
                    'weight': 0.8,
                    'metadata': {'example_type': 'code' if has_code else 'text'}
                }
        
        return None
    
    def _detect_question_answer(self, source: Dict, target: Dict, 
                               source_idx: int, target_idx: int) -> Optional[Dict]:
        """Detect question-answer relationships"""
        source_content = source.get('content', '')
        target_content = target.get('content', '')
        
        # Check if source ends with question
        if source_content.strip().endswith('?'):
            # Check if target is adjacent and from different speaker
            source_speaker = source.get('metadata', {}).get('speaker')
            target_speaker = target.get('metadata', {}).get('speaker')
            
            if target_idx == source_idx + 1 and source_speaker != target_speaker:
                return {
                    'source': source['id'],
                    'target': target['id'],
                    'weight': 0.9,
                    'metadata': {'relationship': 'question_answer'}
                }
        
        return None
    
    def _detect_summary(self, source: Dict, target: Dict, 
                       source_idx: int, target_idx: int) -> Optional[Dict]:
        """Detect summary relationships"""
        target_content = target.get('content', '').lower()
        
        summary_markers = [
            'in summary',
            'to summarize',
            'in conclusion',
            'to conclude',
            'overall',
            'in short',
            'the key points',
            'the main ideas'
        ]
        
        for marker in summary_markers:
            if marker in target_content:
                return {
                    'source': source['id'],
                    'target': target['id'],
                    'weight': 0.7,
                    'metadata': {'summary_marker': marker}
                }
        
        return None
    
    def _detect_code_explanation(self, source: Dict, target: Dict, 
                                source_idx: int, target_idx: int) -> Optional[Dict]:
        """Detect code-explanation relationships"""
        source_has_code = '```' in source.get('content', '')
        target_has_code = '```' in target.get('content', '')
        
        # Code followed by explanation
        if source_has_code and not target_has_code and target_idx == source_idx + 1:
            explanation_words = ['this', 'the above', 'this code', 'this function', 'this class']
            if any(word in target.get('content', '').lower() for word in explanation_words):
                return {
                    'source': source['id'],
                    'target': target['id'],
                    'weight': 0.8,
                    'metadata': {'relationship': 'code_explanation'}
                }
        
        # Explanation followed by code
        if not source_has_code and target_has_code and target_idx == source_idx + 1:
            intro_words = ['following', 'here\'s', 'example', 'implementation', 'code']
            if any(word in source.get('content', '').lower() for word in intro_words):
                return {
                    'source': source['id'],
                    'target': target['id'],
                    'weight': 0.8,
                    'metadata': {'relationship': 'explanation_code'}
                }
        
        return None
    
    def _detect_semantic_similarity(self, source: Dict, target: Dict, 
                                   source_idx: int, target_idx: int) -> Optional[Dict]:
        """Detect semantic similarity using simple word overlap"""
        if abs(source_idx - target_idx) > 5:  # Only check nodes that aren't too far apart
            return None
            
        source_words = set(source.get('content', '').lower().split())
        target_words = set(target.get('content', '').lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                       'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'after',
                       'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'}
        
        source_words = source_words - common_words
        target_words = target_words - common_words
        
        # Calculate Jaccard similarity
        if source_words and target_words:
            intersection = len(source_words & target_words)
            union = len(source_words | target_words)
            similarity = intersection / union
            
            if similarity > 0.3:  # Threshold for semantic similarity
                return {
                    'source': source['id'],
                    'target': target['id'],
                    'weight': similarity,
                    'metadata': {'similarity_score': similarity}
                }
        
        return None
    
    def _add_sequential_edges(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add basic sequential edges for connectivity"""
        edges = []
        
        for i in range(len(nodes) - 1):
            edges.append({
                'source': nodes[i]['id'],
                'target': nodes[i + 1]['id'],
                'weight': 0.5,
                'type': 'sequential',
                'metadata': {'position': 'adjacent'}
            })
        
        return edges
    
    def _deduplicate_edges(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate edges, keeping highest weight"""
        edge_map = {}
        
        for edge in edges:
            key = (edge['source'], edge['target'])
            
            if key not in edge_map or edge['weight'] > edge_map[key]['weight']:
                edge_map[key] = edge
        
        return list(edge_map.values())
    
    def analyze_edge_distribution(self, edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution of edge types"""
        type_counts = defaultdict(int)
        total_weight = 0
        
        for edge in edges:
            type_counts[edge.get('type', 'unknown')] += 1
            total_weight += edge.get('weight', 0)
        
        return {
            'type_distribution': dict(type_counts),
            'total_edges': len(edges),
            'average_weight': total_weight / len(edges) if edges else 0,
            'edge_types': list(type_counts.keys())
        }