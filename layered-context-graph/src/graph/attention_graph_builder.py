"""
Attention Graph Builder Module
----------------------------
This module builds knowledge graphs from attention patterns extracted from transformer models.
It implements the approach described in the condensed architecture document for creating
meaningful relationships between text segments based on attention mechanisms.
"""

import re
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any

class AttentionGraphBuilder:
    """
    Builds knowledge graphs from attention patterns extracted from transformer models
    """
    
    def __init__(self, similarity_threshold: float = 0.7, attention_extractor=None):
        """
        Initialize the attention graph builder
        
        Args:
            similarity_threshold: Threshold for considering segments similar (0.0-1.0)
            attention_extractor: Optional EnhancedAttentionExtractor for real attention patterns
        """
        self.similarity_threshold = similarity_threshold
        self.attention_extractor = attention_extractor
        
        # Try to use attention-based edge detector first
        try:
            from .attention_based_edge_detector import AttentionBasedEdgeDetector
            self.edge_detector = AttentionBasedEdgeDetector(attention_extractor)
            self.use_attention_edges = True
        except ImportError:
            # Fall back to enhanced edge detector
            try:
                from .enhanced_edge_detector import EnhancedEdgeDetector
                self.edge_detector = EnhancedEdgeDetector()
                self.use_attention_edges = False
            except ImportError:
                self.edge_detector = None
                self.use_attention_edges = False
    
    def build_from_attention(self, 
                           attention_data: Dict[str, Any], 
                           segments: List[str]) -> Dict[str, Any]:
        """
        Build knowledge graph from attention patterns
        
        Args:
            attention_data: Dictionary containing attention weights and token mappings
            segments: List of text segments to connect
            
        Returns:
            Dictionary representing the knowledge graph with nodes and edges
        """
        if not attention_data or 'attention' not in attention_data or not segments:
            # Fall back to simple sequential graph
            return self._build_sequential_graph(segments)
        
        graph = {
            'nodes': self._create_nodes(segments),
            'edges': []
        }
        
        # Extract attention weights
        attention_weights = attention_data['attention']
        
        # Create attention-based edges
        graph['edges'] = self._create_attention_based_edges(
            segments, attention_weights, attention_data.get('tokens', [])
        )
        
        # Add enhanced edge detection if available
        if self.edge_detector and isinstance(graph['nodes'], list):
            if self.use_attention_edges and hasattr(self.edge_detector, 'detect_edges_from_attention'):
                # Use attention-based edge detection
                enhanced_edges = self.edge_detector.detect_edges_from_attention(
                    graph['nodes'], 
                    attention_data
                )
            else:
                # Use rule-based edge detection
                enhanced_edges = self.edge_detector.detect_edges(graph['nodes'])
                
            if enhanced_edges:
                graph['edges'].extend(enhanced_edges)
                # Remove duplicates
                seen = set()
                unique_edges = []
                for edge in graph['edges']:
                    key = (edge.get('source'), edge.get('target'), edge.get('type', 'default'))
                    if key not in seen:
                        seen.add(key)
                        unique_edges.append(edge)
                graph['edges'] = unique_edges
        
        # Add classification to nodes
        graph = self._classify_nodes(graph)
        
        return graph
    
    def _create_nodes(self, segments: List[str]) -> List[Dict[str, Any]]:
        """Create graph nodes from text segments"""
        nodes = []
        
        for i, segment in enumerate(segments):
            # Create node with basic metadata
            node = {
                'id': i,
                'content': segment,
                'size': len(segment.split()),
                'type': 'text_segment'
            }
            
            # Add basic content analysis
            node.update(self._analyze_segment_content(segment))
            
            nodes.append(node)
            
        return nodes
    
    def _analyze_segment_content(self, segment: str) -> Dict[str, Any]:
        """Analyze segment content for metadata"""
        # Simple heuristics to categorize content
        metadata = {
            'has_code': self._contains_code(segment),
            'has_math': self._contains_math(segment),
            'is_dialogue': self._is_dialogue(segment),
            'is_technical': self._is_technical(segment),
            'keywords': self._extract_keywords(segment)
        }
        
        return metadata
    
    def _contains_code(self, text: str) -> bool:
        """Check if text contains code snippets"""
        code_markers = ['```', '`', 'def ', 'class ', 'import ', 'from ', '#!/', '</', '/>', '{}', '[]', '()']
        return any(marker in text for marker in code_markers)
    
    def _contains_math(self, text: str) -> bool:
        """Check if text contains mathematical notation"""
        math_markers = ['\\(', '\\)', '\\[', '\\]', '$', '\\sum', '\\int', '\\frac', '\\alpha', '\\beta', '\\gamma']
        return any(marker in text for marker in math_markers)
    
    def _is_dialogue(self, text: str) -> bool:
        """Check if text contains dialogue"""
        # Look for patterns like "Person: Text" or text in quotes
        dialogue_patterns = [
            r'[A-Z][a-z]+\s*:', # Name followed by colon
            r'"[^"]+"\s*,?\s*(?:said|replied|asked)',  # Quoted text with dialogue tag
            r"'[^']+'\s*,?\s*(?:said|replied|asked)"   # Single-quoted with dialogue tag
        ]
        
        return any(re.search(pattern, text) for pattern in dialogue_patterns)
    
    def _is_technical(self, text: str) -> bool:
        """Check if text is technical/scientific"""
        technical_words = [
            'algorithm', 'function', 'method', 'class', 'instance', 'object',
            'parameter', 'variable', 'constant', 'equation', 'theorem', 'proof',
            'implementation', 'framework', 'architecture', 'structure', 'system',
            'process', 'technique', 'analysis', 'evaluation', 'performance'
        ]
        
        # Count occurrences of technical terms
        text_lower = text.lower()
        count = sum(1 for word in technical_words if word in text_lower)
        
        # Consider technical if density of technical terms is high enough
        return count >= 2 or (count > 0 and len(text.split()) < 50)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract potential keywords from text"""
        # Very simple keyword extraction - remove stop words and take frequent terms
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                      'with', 'by', 'for', 'is', 'in', 'to', 'from', 'on', 'at', 'of'}
        
        # Extract words, filter stop words and short words
        words = re.findall(r'\b[a-zA-Z][a-zA-Z]+\b', text.lower())
        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Count occurrences
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Take top 5 keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, count in sorted_words[:5]]
        
        return keywords
    
    def _create_attention_based_edges(self, 
                                    segments: List[str], 
                                    attention_weights: Any,
                                    tokens: List[str]) -> List[Dict[str, Any]]:
        """Create edges based on attention patterns"""
        edges = []
        
        # Add sequential edges first (guaranteed connectivity)
        for i in range(len(segments) - 1):
            edges.append({
                'source': i,
                'target': i + 1,
                'weight': 1.0,
                'type': 'sequential'
            })
        
        # Create additional edges based on attention patterns if available
        try:
            # Add attention-based edges
            attention_edges = self._extract_attention_edges(segments, attention_weights, tokens)
            edges.extend(attention_edges)
        except Exception as e:
            # Fall back to content-based edges if attention processing fails
            content_edges = self._create_content_based_edges(segments)
            edges.extend(content_edges)
        
        return edges
    
    def _extract_attention_edges(self, 
                               segments: List[str],
                               attention_weights: Any,
                               tokens: List[str]) -> List[Dict[str, Any]]:
        """Extract edges from attention patterns"""
        edges = []
        
        # This is a simplified implementation
        # A full implementation would map attention patterns to segments
        
        # Create a mapping of tokens to segments
        segment_tokens = []
        for segment in segments:
            # This is approximate - we don't have exact token mappings
            segment_tokens.append(segment.split())
        
        # Create edges between non-adjacent segments with high attention
        for i in range(len(segments)):
            for j in range(len(segments)):
                if abs(i - j) <= 1:
                    continue  # Skip adjacent segments (already have sequential edges)
                    
                similarity = self._calculate_segment_similarity(segments[i], segments[j])
                if similarity > self.similarity_threshold:
                    edges.append({
                        'source': i,
                        'target': j,
                        'weight': similarity,
                        'type': 'attention'
                    })
        
        return edges
    
    def _create_content_based_edges(self, segments: List[str]) -> List[Dict[str, Any]]:
        """Create edges based on content similarity as fallback"""
        edges = []
        
        # Compare all pairs of segments
        for i in range(len(segments)):
            for j in range(i + 2, len(segments)):  # Skip adjacent (i+1)
                similarity = self._calculate_segment_similarity(segments[i], segments[j])
                if similarity > self.similarity_threshold:
                    edges.append({
                        'source': i,
                        'target': j,
                        'weight': similarity,
                        'type': 'content'
                    })
        
        return edges
    
    def _calculate_segment_similarity(self, seg1: str, seg2: str) -> float:
        """Calculate semantic similarity between segments"""
        # Simple word overlap metric for now
        # In a real implementation, would use embeddings
        words1 = set(re.findall(r'\b\w+\b', seg1.lower()))
        words2 = set(re.findall(r'\b\w+\b', seg2.lower()))
        
        if not words1 or not words2:
            return 0.0
            
        # Jaccard similarity
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return overlap / union if union > 0 else 0.0
    
    def _build_sequential_graph(self, segments: List[str]) -> Dict[str, Any]:
        """Build a simple sequential graph as fallback"""
        nodes = self._create_nodes(segments)
        
        edges = []
        for i in range(len(segments) - 1):
            edges.append({
                'source': i,
                'target': i + 1,
                'weight': 1.0,
                'type': 'sequential'
            })
        
        return {
            'nodes': nodes,
            'edges': edges
        }
    
    def _classify_nodes(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Classify nodes as KEEP, DELETE, or TRACK"""
        if not graph or 'nodes' not in graph:
            return graph
            
        for node in graph['nodes']:
            # Default to KEEP
            node['classification'] = 'KEEP'
            
            content = node.get('content', '')
            
            # Rules for classification
            if len(content.split()) < 20 and not node.get('has_code', False) and not node.get('has_math', False):
                # Very short content without code/math might be DELETE
                if not any(kw in content.lower() for kw in ['important', 'key', 'critical', 'essential']):
                    node['classification'] = 'DELETE'
            
            # Mark nodes with open questions or action items as TRACK
            if re.search(r'\?|TODO|FIXME|NOTE|action item|follow up', content):
                node['classification'] = 'TRACK'
            
            # Always KEEP code, math, and technical content
            if node.get('has_code', False) or node.get('has_math', False) or node.get('is_technical', False):
                node['classification'] = 'KEEP'
        
        return graph
    
    def apply_rule_to_graph(self, graph: Dict[str, Any], rule: str) -> Dict[str, Any]:
        """Apply a natural language rule to reorganize the graph"""
        if not graph or 'nodes' not in graph or not rule:
            return graph
            
        # Different rules trigger different reorganization strategies
        rule_lower = rule.lower()
        
        if 'chronological' in rule_lower or 'time' in rule_lower:
            return self._organize_chronologically(graph)
            
        elif 'theme' in rule_lower or 'topic' in rule_lower:
            return self._organize_by_theme(graph)
            
        elif 'importance' in rule_lower or 'priority' in rule_lower:
            return self._organize_by_importance(graph)
        
        # Default: no change
        return graph
    
    def _organize_chronologically(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Organize graph chronologically"""
        # Implementation would analyze temporal indicators
        # and reorder nodes accordingly
        return graph
    
    def _organize_by_theme(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Organize graph by theme/topic"""
        # Implementation would cluster nodes by similarity
        # and organize hierarchically by theme
        return graph
    
    def _organize_by_importance(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Organize graph by importance/priority"""
        # Implementation would score nodes by importance
        # and reorganize with important content first
        return graph
