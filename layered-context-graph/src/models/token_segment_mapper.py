#!/usr/bin/env python3
"""
Token-Segment Mapper
====================
Efficient mapping between tokens and segments for dual-level processing.
Handles alignment between tokenizer output and document segments.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TokenSegmentAlignment:
    """
    Stores alignment information between tokens and segments.
    """
    token_to_segment: torch.Tensor  # [seq_len] -> segment_id
    segment_to_tokens: Dict[int, List[int]]  # segment_id -> [token_indices]
    segment_boundaries: List[Tuple[int, int]]  # [(start, end) for each segment]
    num_segments: int
    num_tokens: int
    
    def get_segment_mask(self, segment_id: int) -> torch.Tensor:
        """Get boolean mask for tokens in a segment."""
        return self.token_to_segment == segment_id


class TokenSegmentMapper:
    """
    Maps between tokens and segments efficiently.
    Handles various tokenization schemes and segment definitions.
    """
    
    def __init__(self, tokenizer_type: str = 'auto'):
        """
        Initialize mapper.
        
        Args:
            tokenizer_type: Type of tokenizer ('auto', 'wordpiece', 'sentencepiece', 'bpe')
        """
        self.tokenizer_type = tokenizer_type
        self.cache = {}
    
    def create_mapping(self, tokens: List[str], 
                      segments: List[Dict[str, Any]],
                      text: Optional[str] = None) -> TokenSegmentAlignment:
        """
        Create token-to-segment mapping.
        
        Args:
            tokens: List of tokens from tokenizer
            segments: List of segment dictionaries with 'start_pos' and 'end_pos'
            text: Optional original text for position-based mapping
            
        Returns:
            TokenSegmentAlignment object
        """
        num_tokens = len(tokens)
        num_segments = len(segments)
        
        # Initialize mapping
        token_to_segment = torch.zeros(num_tokens, dtype=torch.long)
        segment_to_tokens = defaultdict(list)
        segment_boundaries = []
        
        if text:
            # Position-based mapping
            token_positions = self._get_token_positions(tokens, text)
            
            for seg_idx, segment in enumerate(segments):
                seg_start = segment.get('start_pos', 0)
                seg_end = segment.get('end_pos', len(text))
                
                # Find tokens within segment boundaries
                start_token_idx = None
                end_token_idx = None
                
                for tok_idx, (tok_start, tok_end) in enumerate(token_positions):
                    # Token overlaps with segment
                    if tok_start < seg_end and tok_end > seg_start:
                        if start_token_idx is None:
                            start_token_idx = tok_idx
                        end_token_idx = tok_idx + 1
                        
                        # Assign token to segment
                        token_to_segment[tok_idx] = seg_idx
                        segment_to_tokens[seg_idx].append(tok_idx)
                
                if start_token_idx is not None:
                    segment_boundaries.append((start_token_idx, end_token_idx))
                else:
                    segment_boundaries.append((0, 0))  # Empty segment
        
        else:
            # Fall back to uniform distribution
            tokens_per_segment = num_tokens // num_segments
            remainder = num_tokens % num_segments
            
            current_token = 0
            for seg_idx in range(num_segments):
                # Distribute tokens evenly with remainder
                segment_size = tokens_per_segment + (1 if seg_idx < remainder else 0)
                end_token = min(current_token + segment_size, num_tokens)
                
                segment_boundaries.append((current_token, end_token))
                
                for tok_idx in range(current_token, end_token):
                    token_to_segment[tok_idx] = seg_idx
                    segment_to_tokens[seg_idx].append(tok_idx)
                
                current_token = end_token
        
        return TokenSegmentAlignment(
            token_to_segment=token_to_segment,
            segment_to_tokens=dict(segment_to_tokens),
            segment_boundaries=segment_boundaries,
            num_segments=num_segments,
            num_tokens=num_tokens
        )
    
    def _get_token_positions(self, tokens: List[str], text: str) -> List[Tuple[int, int]]:
        """
        Find character positions of tokens in text.
        
        Args:
            tokens: List of tokens
            text: Original text
            
        Returns:
            List of (start, end) positions for each token
        """
        positions = []
        current_pos = 0
        
        for token in tokens:
            # Handle special tokens
            if token.startswith('[') and token.endswith(']'):
                positions.append((current_pos, current_pos))
                continue
            
            # Clean token based on tokenizer type
            clean_token = self._clean_token(token)
            
            # Find token in text
            start = text.find(clean_token, current_pos)
            if start != -1:
                end = start + len(clean_token)
                positions.append((start, end))
                current_pos = end
            else:
                # Token not found, use current position
                positions.append((current_pos, current_pos))
        
        return positions
    
    def _clean_token(self, token: str) -> str:
        """Clean tokenizer artifacts from token."""
        if self.tokenizer_type == 'wordpiece':
            return token.replace('##', '')
        elif self.tokenizer_type == 'sentencepiece':
            return token.replace('▁', ' ')
        elif self.tokenizer_type == 'bpe':
            return token.replace('Ġ', ' ')
        else:
            # Auto-detect and clean
            cleaned = token
            for artifact in ['##', '▁', 'Ġ']:
                cleaned = cleaned.replace(artifact, ' ' if artifact != '##' else '')
            return cleaned.strip() or token
    
    def create_windowed_mapping(self, window_tokens: List[str],
                              window_start: int,
                              global_alignment: TokenSegmentAlignment) -> torch.Tensor:
        """
        Create token-to-segment mapping for a window.
        
        Args:
            window_tokens: Tokens in current window
            window_start: Start token index in global sequence
            global_alignment: Global token-segment alignment
            
        Returns:
            Local token-to-segment mapping for window
        """
        window_size = len(window_tokens)
        window_end = window_start + window_size
        
        # Extract relevant portion from global mapping
        if window_end <= global_alignment.num_tokens:
            return global_alignment.token_to_segment[window_start:window_end]
        else:
            # Handle edge case where window extends beyond tokens
            valid_size = global_alignment.num_tokens - window_start
            mapping = torch.zeros(window_size, dtype=torch.long)
            if valid_size > 0:
                mapping[:valid_size] = global_alignment.token_to_segment[window_start:]
            return mapping


class HierarchicalTokenMapper:
    """
    Maps tokens through hierarchical segment structures.
    """
    
    def __init__(self, hierarchy_levels: int = 3):
        """
        Initialize hierarchical mapper.
        
        Args:
            hierarchy_levels: Number of hierarchy levels
        """
        self.hierarchy_levels = hierarchy_levels
        self.level_alignments = {}
    
    def create_hierarchical_mapping(self, tokens: List[str],
                                  segment_hierarchy: Dict[int, List[Dict]]) -> Dict[int, TokenSegmentAlignment]:
        """
        Create mappings for each hierarchy level.
        
        Args:
            tokens: List of tokens
            segment_hierarchy: Dict mapping level -> segments at that level
            
        Returns:
            Dict mapping level -> TokenSegmentAlignment
        """
        mapper = TokenSegmentMapper()
        level_alignments = {}
        
        for level, segments in segment_hierarchy.items():
            alignment = mapper.create_mapping(tokens, segments)
            level_alignments[level] = alignment
        
        self.level_alignments = level_alignments
        return level_alignments
    
    def get_multi_level_mapping(self, token_idx: int) -> Dict[int, int]:
        """
        Get segment IDs at all levels for a token.
        
        Args:
            token_idx: Token index
            
        Returns:
            Dict mapping level -> segment_id
        """
        multi_level = {}
        
        for level, alignment in self.level_alignments.items():
            if token_idx < alignment.num_tokens:
                multi_level[level] = int(alignment.token_to_segment[token_idx])
        
        return multi_level


class DynamicSegmentTracker:
    """
    Tracks segment assignments dynamically during processing.
    Useful for streaming or adaptive segmentation.
    """
    
    def __init__(self, initial_segments: Optional[List[Dict]] = None):
        """
        Initialize tracker.
        
        Args:
            initial_segments: Optional initial segment definitions
        """
        self.segments = initial_segments or []
        self.token_count = 0
        self.current_alignment = None
    
    def add_tokens(self, new_tokens: List[str], 
                   segment_decisions: Optional[List[int]] = None):
        """
        Add new tokens and optionally assign to segments.
        
        Args:
            new_tokens: New tokens to add
            segment_decisions: Optional segment assignments
        """
        num_new_tokens = len(new_tokens)
        start_idx = self.token_count
        
        if segment_decisions is None:
            # Auto-assign to current or new segment
            if not self.segments:
                self.segments.append({
                    'id': 0,
                    'start_token': 0,
                    'end_token': num_new_tokens
                })
            else:
                # Extend last segment
                self.segments[-1]['end_token'] = start_idx + num_new_tokens
        else:
            # Use provided decisions
            for i, seg_id in enumerate(segment_decisions):
                token_idx = start_idx + i
                
                # Ensure segment exists
                while seg_id >= len(self.segments):
                    self.segments.append({
                        'id': len(self.segments),
                        'start_token': token_idx,
                        'end_token': token_idx + 1
                    })
                
                # Update segment boundaries
                segment = self.segments[seg_id]
                segment['start_token'] = min(segment.get('start_token', token_idx), token_idx)
                segment['end_token'] = max(segment.get('end_token', token_idx + 1), token_idx + 1)
        
        self.token_count += num_new_tokens
    
    def finalize_segments(self):
        """
        Finalize segment definitions and create alignment.
        
        Returns:
            TokenSegmentAlignment
        """
        # Create token-to-segment mapping
        token_to_segment = torch.zeros(self.token_count, dtype=torch.long)
        segment_to_tokens = defaultdict(list)
        segment_boundaries = []
        
        for seg_idx, segment in enumerate(self.segments):
            start = segment['start_token']
            end = segment['end_token']
            
            segment_boundaries.append((start, end))
            
            for tok_idx in range(start, end):
                token_to_segment[tok_idx] = seg_idx
                segment_to_tokens[seg_idx].append(tok_idx)
        
        self.current_alignment = TokenSegmentAlignment(
            token_to_segment=token_to_segment,
            segment_to_tokens=dict(segment_to_tokens),
            segment_boundaries=segment_boundaries,
            num_segments=len(self.segments),
            num_tokens=self.token_count
        )
        
        return self.current_alignment


class SegmentGraphMapper:
    """
    Maps segments to graph nodes with adjacency information.
    """
    
    def __init__(self):
        """Initialize graph mapper."""
        self.node_to_segment = {}
        self.segment_to_node = {}
        self.adjacency = None
        self.edge_types = None
    
    def create_graph_mapping(self, segments: List[Dict[str, Any]],
                           graph_adjacency: np.ndarray,
                           edge_types: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Create mapping between segments and graph nodes.
        
        Args:
            segments: List of segments with IDs
            graph_adjacency: Adjacency matrix
            edge_types: Optional edge type matrix
            
        Returns:
            Graph mapping information
        """
        # Create bidirectional mapping
        for idx, segment in enumerate(segments):
            seg_id = segment.get('id', str(idx))
            self.node_to_segment[idx] = seg_id
            self.segment_to_node[seg_id] = idx
        
        self.adjacency = torch.from_numpy(graph_adjacency).float()
        if edge_types is not None:
            self.edge_types = torch.from_numpy(edge_types).long()
        
        return {
            'node_to_segment': self.node_to_segment,
            'segment_to_node': self.segment_to_node,
            'num_nodes': len(segments),
            'adjacency': self.adjacency,
            'edge_types': self.edge_types
        }
    
    def get_node_adjacency(self, segment_ids: List[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get adjacency matrix for a subset of segments.
        
        Args:
            segment_ids: List of segment IDs
            
        Returns:
            Tuple of (adjacency_matrix, edge_types) for subset
        """
        # Map segment IDs to node indices
        node_indices = [self.segment_to_node[seg_id] for seg_id in segment_ids 
                       if seg_id in self.segment_to_node]
        
        if not node_indices:
            return torch.zeros(0, 0), None
        
        # Extract subgraph
        n = len(node_indices)
        sub_adjacency = torch.zeros(n, n)
        sub_edge_types = torch.zeros(n, n, dtype=torch.long) if self.edge_types is not None else None
        
        for i, node_i in enumerate(node_indices):
            for j, node_j in enumerate(node_indices):
                sub_adjacency[i, j] = self.adjacency[node_i, node_j]
                if sub_edge_types is not None:
                    sub_edge_types[i, j] = self.edge_types[node_i, node_j]
        
        return sub_adjacency, sub_edge_types