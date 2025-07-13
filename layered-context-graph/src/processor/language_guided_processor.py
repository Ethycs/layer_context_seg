"""
Natural Language Guided Processor Module
--------------------------------------
This module implements the complete natural language guided processing system
as described in the condensed architecture document. It combines instruction seeding,
percolation-optimized context windows, and attention-based graph building to create
a powerful system for processing and reorganizing long-context documents.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import torch
import numpy as np

# Import components from the layered-context-graph system
from ..models.instruction_seeder import InstructionSeeder
from ..models.percolation_context_window import PercolationContextWindow
from ..models.attention_extractor import EnhancedAttentionExtractor
from ..graph.attention_graph_builder import AttentionGraphBuilder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageGuidedProcessor:
    """
    Complete language-guided processor implementing the condensed architecture
    """
    
    def __init__(self, 
                model_source: str = "distilbert-base-uncased", 
                model_type: str = "transformer",
                window_size: int = 8192,
                overlap_ratio: float = 0.25,
                similarity_threshold: float = 0.7,
                instruction_density: float = 0.1):
        """
        Initialize the language-guided processor
        
        Args:
            model_source: Name or path of the model to use
            model_type: Type of model ('transformer' or 'ollama')
            window_size: Maximum window size in words/tokens
            overlap_ratio: Percolation-optimized overlap ratio (0.15-0.30 recommended)
            similarity_threshold: Threshold for considering segments similar
            instruction_density: Density of instructions inserted into text
        """
        self.model_type = model_type
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.similarity_threshold = similarity_threshold
        self.instruction_density = instruction_density
        
        # Initialize components
        self.instruction_seeder = InstructionSeeder()
        self.context_window = PercolationContextWindow(
            size=window_size, 
            overlap_ratio=overlap_ratio
        )
        self.attention_extractor = EnhancedAttentionExtractor(
            model_source=model_source,
            model_type=model_type
        )
        self.graph_builder = AttentionGraphBuilder(
            similarity_threshold=similarity_threshold
        )
        
        # Track discovered head specializations
        self.head_skills = {}
        self.boundary_heads = []
        self.relation_heads = []
        self.clustering_heads = []
        
        # Check for CUDA availability for GPU acceleration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            logger.info(f"Using GPU acceleration: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("GPU not available, using CPU")
    
    def process_document(self, 
                       document: str, 
                       rules: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process a document using natural language rules
        
        Args:
            document: The text document to process
            rules: Dictionary with 'segmentation' and 'reorganization' rules
                  If None, default rules will be used
                  
        Returns:
            Dictionary with processing results including segments, graph, and outputs
        """
        if not document or not document.strip():
            return {'segments': [], 'graph': None, 'outputs': {}}
            
        # 1. Use default rules if none provided
        if not rules:
            rules = {
                'segmentation': "Split at natural topic boundaries",
                'reorganization': "Group by theme and importance"
            }
        
        logger.info(f"Processing document with rules: {rules}")
        
        # 2. Create rule-conditioned input with instruction seeding
        seeded_text = self.instruction_seeder.seed_with_rules(document, rules)
        logger.info("Created instruction-seeded text")
        
        # 3. Create percolation-optimized context windows
        windows = self.context_window.create_window(seeded_text)
        window_graph = self.context_window.get_graph()
        logger.info(f"Created {len(windows)} percolation-optimized windows")
        
        # 4. Process each window with attention extraction
        all_attention_data = []
        for i, window in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}")
            attention_data = self.attention_extractor.extract_attention_patterns(window)
            all_attention_data.append(attention_data)
        
        # 5. Apply natural language rules to guide segmentation
        segmentation_rule = rules.get('segmentation', "Split at natural topic boundaries")
        boundaries = self._detect_rule_guided_boundaries(windows, all_attention_data, segmentation_rule)
        segments = self._create_segments_from_boundaries(document, boundaries)
        logger.info(f"Created {len(segments)} segments based on rule-guided boundaries")
        
        # 6. Build knowledge graph from segments and attention patterns
        graph = self.graph_builder.build_from_attention(
            self._combine_attention_data(all_attention_data),
            segments
        )
        logger.info(f"Built knowledge graph with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges")
        
        # 7. Apply reorganization rule to graph
        reorganization_rule = rules.get('reorganization', "Group by theme and importance")
        organized_graph = self.graph_builder.apply_rule_to_graph(graph, reorganization_rule)
        
        # 8. Generate multiple output formats
        outputs = self._generate_outputs(segments, organized_graph)
        logger.info(f"Generated {len(outputs)} output formats")
        
        return {
            'segments': segments,
            'graph': organized_graph,
            'outputs': outputs,
            'windows': windows,
            'window_graph': window_graph,
        }
    
    def _detect_rule_guided_boundaries(self, 
                                    windows: List[str], 
                                    attention_data: List[Dict[str, Any]],
                                    rule: str) -> List[Tuple[int, float]]:
        """Detect boundaries guided by natural language rules"""
        all_boundaries = []
        
        for i, (window, attn_data) in enumerate(zip(windows, attention_data)):
            # Apply rule to guide attention
            guided_data = self.attention_extractor.apply_natural_language_rule(window, rule)
            
            # Detect boundaries in the guided attention
            window_boundaries = self.attention_extractor.detect_boundaries(guided_data)
            
            # Adjust boundary positions to account for window position in document
            offset = sum(len(w.split()) for w in windows[:i])
            adjusted_boundaries = [(pos + offset, score) for pos, score in window_boundaries]
            
            all_boundaries.extend(adjusted_boundaries)
        
        # Remove duplicate or very close boundaries
        filtered_boundaries = []
        min_distance = 50  # Minimum distance between boundaries
        
        sorted_boundaries = sorted(all_boundaries, key=lambda x: x[0])
        if sorted_boundaries:
            filtered_boundaries.append(sorted_boundaries[0])
            
            for boundary in sorted_boundaries[1:]:
                if boundary[0] - filtered_boundaries[-1][0] >= min_distance:
                    filtered_boundaries.append(boundary)
        
        return filtered_boundaries
    
    def _create_segments_from_boundaries(self, 
                                      document: str, 
                                      boundaries: List[Tuple[int, float]]) -> List[str]:
        """Create text segments based on detected boundaries"""
        if not boundaries:
            return [document]  # Return whole document if no boundaries
            
        # Convert token positions to approximate character positions
        words = document.split()
        char_positions = [0]  # Start of document
        pos = 0
        
        for word in words:
            pos += len(word) + 1  # +1 for space
            char_positions.append(pos)
        
        # Create segments
        segments = []
        start_pos = 0
        
        for boundary_idx, _ in boundaries:
            if boundary_idx < len(char_positions):
                end_pos = char_positions[boundary_idx]
                segment = document[start_pos:end_pos].strip()
                if segment:
                    segments.append(segment)
                start_pos = end_pos
        
        # Add final segment
        final_segment = document[start_pos:].strip()
        if final_segment:
            segments.append(final_segment)
            
        # Ensure we have at least one segment
        if not segments:
            segments = [document]
            
        return segments
    
    def _combine_attention_data(self, all_attention_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine attention data from multiple windows"""
        if not all_attention_data:
            return {}
            
        # Simple implementation - just use the first attention data
        # A more sophisticated implementation would combine data from all windows
        return all_attention_data[0]
    
    def _generate_outputs(self, 
                       segments: List[str], 
                       graph: Dict[str, Any]) -> Dict[str, str]:
        """Generate multiple output formats from the processed document"""
        outputs = {}
        
        # Executive summary
        outputs['summary'] = self._generate_summary(segments, graph)
        
        # Thematic analysis
        outputs['themes'] = self._generate_thematic_analysis(segments, graph)
        
        # Chronological report
        outputs['chronological'] = self._generate_chronological_report(segments, graph)
        
        # Action items
        outputs['action_items'] = self._extract_action_items(segments, graph)
        
        # Key insights
        outputs['key_insights'] = self._extract_key_insights(segments, graph)
        
        return outputs
    
    def _generate_summary(self, 
                       segments: List[str], 
                       graph: Dict[str, Any]) -> str:
        """Generate executive summary from segments and graph"""
        if not segments:
            return ""
            
        # Select KEEP nodes that have high importance
        keep_segments = []
        
        if graph and 'nodes' in graph:
            for node in graph['nodes']:
                if node.get('classification') == 'KEEP':
                    keep_segments.append(node.get('content', ''))
        
        if not keep_segments:
            # Fall back to using all segments
            keep_segments = segments
        
        # Join the selected segments
        summary = "\n\n".join(keep_segments)
        
        return summary
    
    def _generate_thematic_analysis(self, 
                                 segments: List[str], 
                                 graph: Dict[str, Any]) -> str:
        """Generate thematic analysis from segments and graph"""
        # In a full implementation, this would group segments by theme
        # For now, just return the segments
        return "\n\n".join(segments)
    
    def _generate_chronological_report(self, 
                                    segments: List[str], 
                                    graph: Dict[str, Any]) -> str:
        """Generate chronological report from segments and graph"""
        # In a full implementation, this would order segments chronologically
        # For now, just return the segments
        return "\n\n".join(segments)
    
    def _extract_action_items(self, 
                           segments: List[str], 
                           graph: Dict[str, Any]) -> str:
        """Extract action items from segments and graph"""
        action_items = []
        
        if graph and 'nodes' in graph:
            for node in graph['nodes']:
                if node.get('classification') == 'TRACK':
                    action_items.append(node.get('content', ''))
        
        if not action_items:
            # Try to extract action items based on text patterns
            for segment in segments:
                if re.search(r'(?i)(todo|action item|follow up|next steps)', segment):
                    action_items.append(segment)
        
        return "\n\n".join(action_items)
    
    def _extract_key_insights(self, 
                           segments: List[str], 
                           graph: Dict[str, Any]) -> str:
        """Extract key insights from segments and graph"""
        insights = []
        
        if graph and 'nodes' in graph:
            for node in graph['nodes']:
                if node.get('classification') == 'KEEP':
                    if re.search(r'(?i)(key|important|critical|insight|finding)', node.get('content', '')):
                        insights.append(node.get('content', ''))
        
        return "\n\n".join(insights)
    
    def discover_head_specializations(self, test_corpus: str) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Discover what each attention head is good at"""
        if not test_corpus or not test_corpus.strip():
            return {}
            
        logger.info("Discovering head specializations...")
        self.head_skills = self.attention_extractor.discover_head_specializations(test_corpus)
        
        # Assign heads to tasks based on skills
        if self.head_skills:
            # Sort heads by boundary detection skill
            boundary_heads = sorted(
                self.head_skills.items(),
                key=lambda x: x[1]['boundary'],
                reverse=True
            )
            self.boundary_heads = [head for head, _ in boundary_heads[:5]]
            
            # Sort heads by relation detection skill
            relation_heads = sorted(
                self.head_skills.items(),
                key=lambda x: x[1]['relation'],
                reverse=True
            )
            self.relation_heads = [head for head, _ in relation_heads[:5]]
            
            # Sort heads by clustering skill
            clustering_heads = sorted(
                self.head_skills.items(),
                key=lambda x: x[1]['cluster'],
                reverse=True
            )
            self.clustering_heads = [head for head, _ in clustering_heads[:5]]
            
            logger.info(f"Discovered specializations for {len(self.head_skills)} heads")
        
        return self.head_skills
    
    def adaptive_head_selection(self, text_type: str) -> List[Tuple[int, int]]:
        """Choose different heads for different text types"""
        if text_type == "code":
            return self.boundary_heads
        elif text_type == "narrative":
            return self.relation_heads
        elif text_type == "technical":
            return self.clustering_heads
        else:
            # Default: use a mix of heads
            return self.boundary_heads[:2] + self.relation_heads[:2] + self.clustering_heads[:1]
    
    def program_with_natural_language(self, 
                                   segmentation_rule: str, 
                                   reorganization_rule: str) -> Dict[str, str]:
        """Program the system with plain English rules"""
        return {
            'segmentation': segmentation_rule,
            'reorganization': reorganization_rule
        }
