#!/usr/bin/env python3
"""
Formatting-Preserving Graph Reassembler
======================================
Reconstructs documents from graphs while preserving original formatting.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import logging

from config.graph_config import ReconstructionConfig
from partitioning.formatting_preserving_partition_manager import Segment

logger = logging.getLogger(__name__)


class FormattingPreservingReassembler:
    """
    Reassembles graph nodes into documents while preserving original formatting.
    """
    
    def __init__(self, config: ReconstructionConfig = None):
        """
        Initialize with configuration.
        
        Args:
            config: ReconstructionConfig instance
        """
        self.config = config or ReconstructionConfig()
        
    def reassemble_graph(self, nodes: List[Dict[str, Any]], 
                        edges: List[Dict[str, Any]], 
                        original_text: str = None) -> Dict[str, Any]:
        """
        Reassemble graph into document preserving formatting.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            original_text: Original text for formatting reference
            
        Returns:
            Dictionary with reassembled document and metadata
        """
        # Get the reassembled text
        reassembled_text = self._reassemble_graph_to_text(nodes, edges, original_text)
        
        # Return in expected format
        return {
            'tape2': reassembled_text,
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'strategy': self.config.default_strategy,
                'formatting_preserved': True
            }
        }
    
    def _reassemble_graph_to_text(self, nodes: List[Dict[str, Any]], 
                                  edges: List[Dict[str, Any]], 
                                  original_text: str = None) -> str:
        """
        Reassemble graph into document preserving formatting.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            original_text: Original text for formatting reference
            
        Returns:
            Reassembled document with formatting preserved
        """
        # Convert nodes to segments if they have formatting info
        segments = self._nodes_to_segments(nodes)
        
        # Determine reassembly order based on strategy
        ordered_segments = self._order_segments(segments, edges)
        
        # Reassemble with formatting
        return self._reassemble_with_formatting(ordered_segments, original_text)
    
    def _nodes_to_segments(self, nodes: List[Dict[str, Any]]) -> List[Segment]:
        """Convert graph nodes to Segment objects"""
        segments = []
        
        for node in nodes:
            # Extract segment information
            content = node.get('content', '')
            formatting = node.get('formatting', {})
            
            # Create segment
            segment = Segment(
                content=content,
                start_pos=node.get('start_pos', 0),
                end_pos=node.get('end_pos', len(content)),
                formatting=formatting,
                segment_type=node.get('segment_type', 'text')
            )
            
            segments.append(segment)
        
        return segments
    
    def _order_segments(self, segments: List[Segment], 
                       edges: List[Dict[str, Any]]) -> List[Segment]:
        """Order segments based on reconstruction strategy"""
        
        if self.config.default_strategy == 'linear':
            # Simple position-based ordering
            return sorted(segments, key=lambda s: s.start_pos)
            
        elif self.config.default_strategy == 'hierarchical':
            # Build hierarchy from edges
            return self._hierarchical_ordering(segments, edges)
            
        elif self.config.default_strategy == 'thematic':
            # Group by themes/topics
            return self._thematic_ordering(segments, edges)
            
        else:
            # Default to linear
            return sorted(segments, key=lambda s: s.start_pos)
    
    def _hierarchical_ordering(self, segments: List[Segment], 
                              edges: List[Dict[str, Any]]) -> List[Segment]:
        """Order segments hierarchically based on graph structure"""
        # Build adjacency map
        adjacency = {}
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            if source and target:
                adjacency.setdefault(source, []).append(target)
        
        # Find root segments (no incoming edges)
        all_targets = set()
        for targets in adjacency.values():
            all_targets.update(targets)
        
        roots = []
        segment_map = {s.start_pos: s for s in segments}
        
        for segment in segments:
            segment_id = str(segment.start_pos)
            if segment_id not in all_targets:
                roots.append(segment)
        
        # Depth-first traversal
        ordered = []
        visited = set()
        
        def traverse(segment):
            segment_id = str(segment.start_pos)
            if segment_id in visited:
                return
            visited.add(segment_id)
            ordered.append(segment)
            
            # Visit children
            for child_id in adjacency.get(segment_id, []):
                if child_id.isdigit():
                    child_pos = int(child_id)
                    if child_pos in segment_map:
                        traverse(segment_map[child_pos])
        
        # Traverse from each root
        for root in roots:
            traverse(root)
        
        # Add any unvisited segments
        for segment in segments:
            if str(segment.start_pos) not in visited:
                ordered.append(segment)
        
        return ordered
    
    def _thematic_ordering(self, segments: List[Segment], 
                          edges: List[Dict[str, Any]]) -> List[Segment]:
        """Order segments by theme/topic clusters"""
        # Group segments by type first
        groups = {}
        for segment in segments:
            seg_type = segment.segment_type
            groups.setdefault(seg_type, []).append(segment)
        
        # Order within groups by position
        ordered = []
        
        # Priority order for segment types
        type_order = ['code_block', 'conversation', 'paragraph', 'text', 
                     'bullet_list', 'numbered_list', 'indented_code']
        
        for seg_type in type_order:
            if seg_type in groups:
                group_segments = sorted(groups[seg_type], key=lambda s: s.start_pos)
                ordered.extend(group_segments)
        
        # Add any remaining types
        for seg_type, group_segments in groups.items():
            if seg_type not in type_order:
                group_segments = sorted(group_segments, key=lambda s: s.start_pos)
                ordered.extend(group_segments)
        
        return ordered
    
    def _reassemble_with_formatting(self, segments: List[Segment], 
                                   original_text: str = None) -> str:
        """Reassemble segments preserving formatting"""
        if not segments:
            return ""
        
        parts = []
        prev_segment = None
        
        for i, segment in enumerate(segments):
            # Extract formatting
            formatting = segment.formatting
            content = segment.content
            
            # Apply section headers if configured
            if self.config.add_section_headers and i > 0:
                if segment.segment_type != prev_segment.segment_type:
                    header = self._generate_section_header(segment.segment_type)
                    if header:
                        parts.append(f"\n\n{header}\n\n")
            
            # Handle spacing between segments
            if prev_segment:
                spacing = self._determine_spacing(prev_segment, segment)
                parts.append(spacing)
            
            # Apply formatting to content
            formatted_content = self._apply_formatting(content, formatting)
            
            # Handle special segment types
            if segment.segment_type == 'code_block':
                if not formatted_content.startswith('```'):
                    language = formatting.get('language', '')
                    formatted_content = f"```{language}\n{formatted_content}\n```"
            
            elif segment.segment_type == 'conversation':
                # Ensure speaker labels are preserved
                if self.config.preserve_speaker_labels:
                    formatted_content = self._format_conversation(formatted_content, formatting)
            
            parts.append(formatted_content)
            prev_segment = segment
        
        # Join and clean up
        result = ''.join(parts)
        
        # Preserve empty lines if configured
        if not self.config.preserve_empty_lines:
            # Remove excessive empty lines
            result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result
    
    def _apply_formatting(self, content: str, formatting: Dict[str, Any]) -> str:
        """Apply formatting metadata to content"""
        
        # Preserve indentation
        if self.config.preserve_indentation and 'indentation' in formatting:
            indent = formatting['indentation']
            lines = content.split('\n')
            indented_lines = []
            for line in lines:
                if line and not line.startswith(indent):
                    line = indent + line
                indented_lines.append(line)
            content = '\n'.join(indented_lines)
        
        # Preserve line breaks
        if not self.config.preserve_original_formatting:
            # Normalize line breaks
            content = re.sub(r'\n+', '\n', content)
        
        return content
    
    def _determine_spacing(self, prev_segment: Segment, 
                          current_segment: Segment) -> str:
        """Determine spacing between segments"""
        
        # Paragraph spacing
        if self.config.maintain_paragraph_spacing:
            if prev_segment.segment_type == 'paragraph' or current_segment.segment_type == 'paragraph':
                return '\n\n'
        
        # Code block spacing
        if 'code' in prev_segment.segment_type or 'code' in current_segment.segment_type:
            return '\n\n'
        
        # Conversation continuity
        if (prev_segment.segment_type == 'conversation' and 
            current_segment.segment_type == 'conversation'):
            return '\n'
        
        # List continuity
        if ('list' in prev_segment.segment_type and 
            'list' in current_segment.segment_type):
            return '\n'
        
        # Default spacing
        return '\n'
    
    def _generate_section_header(self, segment_type: str) -> Optional[str]:
        """Generate section header for segment type"""
        headers = {
            'code_block': '## Code',
            'conversation': '## Conversation',
            'bullet_list': '## Key Points',
            'numbered_list': '## Steps'
        }
        return headers.get(segment_type)
    
    def _format_conversation(self, content: str, formatting: Dict[str, Any]) -> str:
        """Format conversation content"""
        # Ensure speaker labels are clear
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Check if line has speaker label
            if ':' in line and not line.startswith(' '):
                # Ensure consistent speaker format
                parts = line.split(':', 1)
                if len(parts) == 2:
                    speaker = parts[0].strip()
                    text = parts[1].strip()
                    line = f"{speaker}: {text}"
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def reassemble_by_mode(self, nodes: List[Dict[str, Any]], 
                          edges: List[Dict[str, Any]], 
                          mode: str,
                          original_text: str = None) -> str:
        """
        Reassemble based on specific mode.
        
        Args:
            nodes: Graph nodes
            edges: Graph edges  
            mode: Reassembly mode ('timeline', 'speaker', 'evolution', etc.)
            original_text: Original text for reference
            
        Returns:
            Reassembled document
        """
        # Temporarily override strategy
        original_strategy = self.config.default_strategy
        
        mode_strategies = {
            'timeline': 'linear',
            'speaker': 'hierarchical',
            'evolution': 'thematic',
            'current_state': 'thematic',
            'research': 'hierarchical'
        }
        
        self.config.default_strategy = mode_strategies.get(mode, 'linear')
        
        # Reassemble with mode-specific strategy
        result = self.reassemble_graph(nodes, edges, original_text)
        
        # Restore original strategy
        self.config.default_strategy = original_strategy
        
        return result