#!/usr/bin/env python3
"""
Conversation-Specific Edge Types
================================
Defines temporal and semantic edge types for conversation tracking.
"""

from enum import Enum
from typing import Dict, List, Tuple, Optional


class ConversationEdgeType(Enum):
    """Edge types for conversation graph relationships"""
    
    # Temporal relationships
    FOLLOWS = "follows"                # Direct temporal sequence
    RESPONDS_TO = "responds_to"        # Direct response to previous statement
    
    # Semantic evolution relationships
    BUILDS_ON = "builds_on"            # Idea B develops idea A
    CONTRADICTS = "contradicts"        # New information conflicts with earlier
    CONFIRMS = "confirms"              # Later statement supports earlier claim
    CLARIFIES = "clarifies"            # Provides clarification
    QUESTIONS = "questions"            # Poses a question about previous content
    ANSWERS = "answers"                # Provides answer to question
    
    # Reference relationships
    REFERENCES = "references"          # Explicit back-reference
    ELABORATES = "elaborates"          # Expands on previous point
    SUMMARIZES = "summarizes"          # Summarizes previous points
    
    # Topic relationships
    SHIFTS_TOPIC = "shifts_topic"      # Changes subject
    RETURNS_TO = "returns_to"          # Returns to earlier topic
    
    # Agreement relationships
    AGREES_WITH = "agrees_with"        # Expresses agreement
    DISAGREES_WITH = "disagrees_with"  # Expresses disagreement
    NEUTRAL_TO = "neutral_to"          # Neither agrees nor disagrees


class ConversationEdgeAnalyzer:
    """Analyzes and classifies edges in conversation graphs using attention patterns"""
    
    def __init__(self, attention_extractor=None):
        self.attention_extractor = attention_extractor
        self.edge_weights = {
            # Stronger connections
            ConversationEdgeType.RESPONDS_TO: 1.0,
            ConversationEdgeType.ANSWERS: 0.95,
            ConversationEdgeType.BUILDS_ON: 0.9,
            ConversationEdgeType.CLARIFIES: 0.85,
            
            # Medium connections
            ConversationEdgeType.CONFIRMS: 0.7,
            ConversationEdgeType.ELABORATES: 0.7,
            ConversationEdgeType.REFERENCES: 0.65,
            ConversationEdgeType.AGREES_WITH: 0.6,
            
            # Weaker connections
            ConversationEdgeType.CONTRADICTS: 0.5,
            ConversationEdgeType.DISAGREES_WITH: 0.5,
            ConversationEdgeType.QUESTIONS: 0.55,
            ConversationEdgeType.SHIFTS_TOPIC: 0.3,
            
            # Default connections
            ConversationEdgeType.FOLLOWS: 0.4,
            ConversationEdgeType.NEUTRAL_TO: 0.35,
            ConversationEdgeType.RETURNS_TO: 0.6,
            ConversationEdgeType.SUMMARIZES: 0.75
        }
    
    def analyze_edge_type(self, source_node: Dict, target_node: Dict, 
                         attention_data: Optional[Dict] = None) -> Tuple[ConversationEdgeType, float]:
        """
        Determine edge type between two nodes using attention patterns and content analysis.
        
        Returns:
            Tuple of (edge_type, confidence_score)
        """
        source_content = source_node.get('content', '')
        target_content = target_node.get('content', '')
        
        # If we have attention data, use it for more sophisticated analysis
        if attention_data and self.attention_extractor:
            return self._analyze_with_attention(source_node, target_node, attention_data)
        
        # Fallback to content-based analysis
        return self._analyze_with_content(source_content, target_content)
    
    def _analyze_with_attention(self, source_node: Dict, target_node: Dict, 
                               attention_data: Dict) -> Tuple[ConversationEdgeType, float]:
        """Use attention patterns to determine relationship type"""
        
        # Extract attention features between nodes
        source_attention = source_node.get('attention', {})
        target_attention = target_node.get('attention', {})
        
        # Calculate attention similarity
        if source_attention and target_attention:
            similarity = self._calculate_attention_similarity(source_attention, target_attention)
            
            # High similarity suggests building/elaborating
            if similarity > 0.8:
                return (ConversationEdgeType.BUILDS_ON, similarity)
            elif similarity > 0.6:
                return (ConversationEdgeType.ELABORATES, similarity)
            elif similarity < 0.3:
                return (ConversationEdgeType.SHIFTS_TOPIC, 1 - similarity)
        
        # Check for question-answer patterns in attention
        if self._is_question_pattern(source_attention) and self._is_answer_pattern(target_attention):
            return (ConversationEdgeType.ANSWERS, 0.9)
        
        # Default to temporal following
        return (ConversationEdgeType.FOLLOWS, 0.5)
    
    def _analyze_with_content(self, source_content: str, target_content: str) -> Tuple[ConversationEdgeType, float]:
        """Fallback content-based analysis"""
        source_lower = source_content.lower()
        target_lower = target_content.lower()
        
        # Question-answer detection
        if '?' in source_content and any(marker in target_lower for marker in ['answer', 'yes', 'no', 'because']):
            return (ConversationEdgeType.ANSWERS, 0.8)
        
        # Agreement/disagreement
        if any(marker in target_lower for marker in ['agree', 'correct', 'exactly', 'yes']):
            return (ConversationEdgeType.AGREES_WITH, 0.7)
        elif any(marker in target_lower for marker in ['disagree', 'no', 'however', 'but']):
            return (ConversationEdgeType.DISAGREES_WITH, 0.7)
        
        # Reference detection
        if any(ref in target_lower for ref in ['as mentioned', 'earlier', 'previously', 'above']):
            return (ConversationEdgeType.REFERENCES, 0.6)
        
        # Topic shift
        if any(marker in target_lower for marker in ['moving on', 'another point', 'by the way']):
            return (ConversationEdgeType.SHIFTS_TOPIC, 0.8)
        
        # Default
        return (ConversationEdgeType.FOLLOWS, 0.4)
    
    def _calculate_attention_similarity(self, attention1: Dict, attention2: Dict) -> float:
        """Calculate similarity between two attention patterns"""
        # Simplified implementation - should use actual attention scores
        scores1 = attention1.get('attention_scores', [])
        scores2 = attention2.get('attention_scores', [])
        
        if not scores1 or not scores2:
            return 0.5
        
        # Calculate cosine similarity or other metric
        # For now, return a placeholder
        return 0.6
    
    def _is_question_pattern(self, attention: Dict) -> bool:
        """Detect if attention pattern suggests a question"""
        # Questions often have rising attention at the end
        scores = attention.get('attention_scores', [])
        if scores and len(scores) > 2:
            return scores[-1] > scores[0]  # Simplified check
        return False
    
    def _is_answer_pattern(self, attention: Dict) -> bool:
        """Detect if attention pattern suggests an answer"""
        # Answers often have strong initial attention
        scores = attention.get('attention_scores', [])
        if scores and len(scores) > 2:
            return scores[0] > scores[-1]  # Simplified check
        return False
    
    def get_edge_weight(self, edge_type: ConversationEdgeType) -> float:
        """Get the weight for a given edge type"""
        return self.edge_weights.get(edge_type, 0.5)
    
    def create_conversation_edges(self, nodes: List[Dict], use_attention: bool = True) -> List[Dict]:
        """
        Create conversation-aware edges between nodes.
        
        Args:
            nodes: List of graph nodes with content and optional attention data
            use_attention: Whether to use attention patterns for analysis
            
        Returns:
            List of edges with conversation-specific types and weights
        """
        edges = []
        
        for i in range(len(nodes) - 1):
            source = nodes[i]
            target = nodes[i + 1]
            
            # Analyze edge type
            edge_type, confidence = self.analyze_edge_type(source, target)
            
            # Create edge
            edge = {
                'source': source['id'],
                'target': target['id'],
                'type': edge_type.value,
                'weight': self.get_edge_weight(edge_type) * confidence,
                'metadata': {
                    'edge_type': edge_type.value,
                    'confidence': confidence,
                    'temporal_distance': 1  # Direct sequence
                }
            }
            edges.append(edge)
            
            # Look for non-sequential relationships (references, returns to topic)
            if i > 0:
                for j in range(i):
                    earlier = nodes[j]
                    later = nodes[i + 1]
                    
                    # Check for references or topic returns
                    ref_type, ref_confidence = self._check_reference_relationship(earlier, later)
                    if ref_type and ref_confidence > 0.5:
                        ref_edge = {
                            'source': earlier['id'],
                            'target': later['id'],
                            'type': ref_type.value,
                            'weight': self.get_edge_weight(ref_type) * ref_confidence,
                            'metadata': {
                                'edge_type': ref_type.value,
                                'confidence': ref_confidence,
                                'temporal_distance': i + 1 - j
                            }
                        }
                        edges.append(ref_edge)
        
        return edges
    
    def _check_reference_relationship(self, earlier_node: Dict, later_node: Dict) -> Tuple[Optional[ConversationEdgeType], float]:
        """Check if later node references or returns to earlier node"""
        earlier_content = earlier_node.get('content', '').lower()
        later_content = later_node.get('content', '').lower()
        
        # Extract key concepts from earlier node
        earlier_words = set(earlier_content.split())
        later_words = set(later_content.split())
        
        # Check for explicit references
        if any(ref in later_content for ref in ['earlier', 'previously', 'as mentioned', 'you said']):
            word_overlap = len(earlier_words & later_words) / len(earlier_words) if earlier_words else 0
            if word_overlap > 0.3:
                return (ConversationEdgeType.REFERENCES, 0.8)
        
        # Check for topic return (high content similarity but temporal distance)
        word_overlap = len(earlier_words & later_words) / len(earlier_words | later_words) if earlier_words or later_words else 0
        if word_overlap > 0.5:
            return (ConversationEdgeType.RETURNS_TO, word_overlap)
        
        return (None, 0.0)