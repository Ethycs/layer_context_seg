#!/usr/bin/env python3
"""
Content-Preserving Partition Manager
====================================
Maintains larger, more meaningful segments while still enabling graph construction
"""

import re
from typing import List, Dict, Any, Tuple


class ContentPreservingPartitionManager:
    """
    Partition manager that preserves more content per node
    """
    
    def __init__(self, 
                 min_segment_length: int = 800,
                 target_segment_length: int = 1500,
                 max_segment_length: int = 3000,
                 overlap_ratio: float = 0.2):
        """
        Initialize with content-preserving parameters
        
        Args:
            min_segment_length: Minimum characters per segment
            target_segment_length: Target segment size
            max_segment_length: Maximum before forced split
            overlap_ratio: Overlap between segments (15-30% for percolation)
        """
        self.min_segment_length = min_segment_length
        self.target_segment_length = target_segment_length
        self.max_segment_length = max_segment_length
        self.overlap_ratio = overlap_ratio
        self.partitions = []
        
    def create_partitions(self, text: str) -> List[Dict[str, Any]]:
        """
        Create partitions that preserve meaningful content chunks
        """
        if not text:
            return []
            
        # First, identify natural boundaries
        natural_segments = self._identify_natural_segments(text)
        
        # Then, apply smart segmentation
        segments = self._smart_segmentation(natural_segments)
        
        # Create partition objects with metadata
        partitions = []
        for i, segment in enumerate(segments):
            partition = {
                'id': f'segment_{i}',
                'content': segment['content'],
                'start_pos': segment['start'],
                'end_pos': segment['end'],
                'segment_type': segment.get('type', 'general'),
                'metadata': {
                    'length': len(segment['content']),
                    'has_code': self._contains_code(segment['content']),
                    'speaker': segment.get('speaker', None),
                    'round': segment.get('round', 0)
                }
            }
            partitions.append(partition)
            
        self.partitions = partitions
        return partitions
    
    def _identify_natural_segments(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify natural content boundaries
        """
        segments = []
        
        # Check if it's a conversation
        if self._is_conversation(text):
            segments = self._segment_conversation(text)
        # Check if it's code-heavy
        elif self._is_code_heavy(text):
            segments = self._segment_code_document(text)
        # Otherwise use general segmentation
        else:
            segments = self._segment_general_text(text)
            
        return segments
    
    def _is_conversation(self, text: str) -> bool:
        """Check if text is a conversation/dialogue"""
        conversation_patterns = [
            r'^(User|Claude|Speaker\s*[A-Z]|Assistant|Human):',
            r'\n(User|Claude|Speaker\s*[A-Z]|Assistant|Human):'
        ]
        
        for pattern in conversation_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False
    
    def _is_code_heavy(self, text: str) -> bool:
        """Check if text contains significant code"""
        code_indicators = ['```', 'def ', 'class ', 'function ', 'import ']
        code_count = sum(text.count(indicator) for indicator in code_indicators)
        return code_count > 5
    
    def _contains_code(self, text: str) -> bool:
        """Check if segment contains code"""
        return '```' in text or 'def ' in text or 'class ' in text
    
    def _segment_conversation(self, text: str) -> List[Dict[str, Any]]:
        """
        Segment conversation preserving full turns
        """
        segments = []
        
        # Split by speaker turns
        turn_pattern = re.compile(r'^(User|Claude|Speaker\s*[A-Z]|Assistant|Human):\s*', re.MULTILINE)
        
        parts = turn_pattern.split(text)
        
        # Reconstruct turns
        current_pos = 0
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                speaker = parts[i]
                content = parts[i + 1].strip()
                
                # Keep full speaker turns together
                full_turn = f"{speaker}: {content}"
                
                segment = {
                    'content': full_turn,
                    'start': current_pos,
                    'end': current_pos + len(full_turn),
                    'type': 'conversation_turn',
                    'speaker': speaker
                }
                
                current_pos = segment['end']
                segments.append(segment)
        
        # Merge short adjacent turns if needed
        merged_segments = self._merge_short_segments(segments)
        
        return merged_segments
    
    def _segment_code_document(self, text: str) -> List[Dict[str, Any]]:
        """
        Segment code-heavy documents preserving code blocks
        """
        segments = []
        current_pos = 0
        
        # Find all code blocks
        code_block_pattern = re.compile(r'```[\s\S]*?```', re.MULTILINE)
        
        last_end = 0
        for match in code_block_pattern.finditer(text):
            # Add text before code block
            if match.start() > last_end:
                pre_text = text[last_end:match.start()].strip()
                if pre_text:
                    segments.append({
                        'content': pre_text,
                        'start': last_end,
                        'end': match.start(),
                        'type': 'narrative'
                    })
            
            # Add code block as single segment
            segments.append({
                'content': match.group(),
                'start': match.start(),
                'end': match.end(),
                'type': 'code_block'
            })
            
            last_end = match.end()
        
        # Add remaining text
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                segments.append({
                    'content': remaining,
                    'start': last_end,
                    'end': len(text),
                    'type': 'narrative'
                })
        
        return segments
    
    def _segment_general_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Segment general text by paragraphs and semantic boundaries
        """
        segments = []
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', text)
        
        current_pos = 0
        current_segment = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_length = len(para)
            
            # If adding this paragraph exceeds max length, save current segment
            if current_length > 0 and current_length + para_length > self.max_segment_length:
                segment_content = '\n\n'.join(current_segment)
                segments.append({
                    'content': segment_content,
                    'start': current_pos,
                    'end': current_pos + len(segment_content),
                    'type': 'general'
                })
                current_pos += len(segment_content) + 2
                current_segment = [para]
                current_length = para_length
            else:
                current_segment.append(para)
                current_length += para_length + 2  # Account for \n\n
                
                # If we've reached target length, consider saving
                if current_length >= self.target_segment_length:
                    # Look for a good breaking point
                    if self._is_good_break_point(para):
                        segment_content = '\n\n'.join(current_segment)
                        segments.append({
                            'content': segment_content,
                            'start': current_pos,
                            'end': current_pos + len(segment_content),
                            'type': 'general'
                        })
                        current_pos += len(segment_content) + 2
                        current_segment = []
                        current_length = 0
        
        # Add remaining content
        if current_segment:
            segment_content = '\n\n'.join(current_segment)
            segments.append({
                'content': segment_content,
                'start': current_pos,
                'end': current_pos + len(segment_content),
                'type': 'general'
            })
        
        return segments
    
    def _is_good_break_point(self, text: str) -> bool:
        """
        Check if this is a good point to break the text
        """
        # Good break indicators
        break_indicators = [
            text.endswith('.'),
            text.endswith('?'),
            text.endswith('!'),
            text.endswith(':'),
            'In summary' in text,
            'To conclude' in text,
            'Therefore' in text,
            text.startswith('#'),  # Markdown headers
            text.startswith('##'),
            text.startswith('###')
        ]
        
        return any(break_indicators)
    
    def _merge_short_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge segments that are too short
        """
        if not segments:
            return segments
            
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            # Merge if current is too short
            if len(current['content']) < self.min_segment_length:
                # Merge with next
                current = {
                    'content': current['content'] + '\n\n' + next_seg['content'],
                    'start': current['start'],
                    'end': next_seg['end'],
                    'type': current.get('type', 'general'),
                    'speaker': current.get('speaker', next_seg.get('speaker'))
                }
            else:
                # Save current and move to next
                merged.append(current)
                current = next_seg
        
        # Don't forget the last segment
        merged.append(current)
        
        return merged
    
    def _smart_segmentation(self, natural_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply smart segmentation to ensure good sizes while preserving content
        """
        final_segments = []
        
        for segment in natural_segments:
            content = segment['content']
            length = len(content)
            
            # If segment is within bounds, keep as is
            if self.min_segment_length <= length <= self.max_segment_length:
                final_segments.append(segment)
            
            # If too long, split intelligently
            elif length > self.max_segment_length:
                sub_segments = self._split_large_segment(segment)
                final_segments.extend(sub_segments)
            
            # If too short, it will be handled by merging later
            else:
                final_segments.append(segment)
        
        # Final merge pass for short segments
        return self._merge_short_segments(final_segments)
    
    def _split_large_segment(self, segment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a large segment intelligently
        """
        content = segment['content']
        segments = []
        
        # Try to split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        current_chunk = []
        current_length = 0
        start_pos = segment['start']
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.target_segment_length and current_chunk:
                # Save current chunk
                chunk_content = ' '.join(current_chunk)
                segments.append({
                    'content': chunk_content,
                    'start': start_pos,
                    'end': start_pos + len(chunk_content),
                    'type': segment.get('type', 'general')
                })
                start_pos += len(chunk_content) + 1
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1
        
        # Add remaining
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            segments.append({
                'content': chunk_content,
                'start': start_pos,
                'end': segment['end'],
                'type': segment.get('type', 'general')
            })
        
        return segments
    
    def create_overlapping_windows(self, partitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create overlapping windows for percolation
        """
        if not partitions or len(partitions) < 2:
            return partitions
            
        overlapped = []
        
        for i in range(len(partitions)):
            current = partitions[i]
            
            # Calculate overlap size
            overlap_chars = int(len(current['content']) * self.overlap_ratio)
            
            # Add overlap from previous segment
            if i > 0:
                prev = partitions[i-1]
                prev_overlap = prev['content'][-overlap_chars:]
                current['content'] = prev_overlap + '\n[OVERLAP]\n' + current['content']
                current['has_prev_overlap'] = True
            
            # Add overlap to next segment
            if i < len(partitions) - 1:
                next_seg = partitions[i+1]
                next_overlap = current['content'][:overlap_chars]
                current['next_overlap'] = next_overlap
            
            overlapped.append(current)
        
        return overlapped