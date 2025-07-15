#!/usr/bin/env python3
"""
Formatting-Preserving Partition Manager
======================================
Maintains original formatting while segmenting documents.
"""

import re
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
from dataclasses import dataclass

from config.graph_config import DisassemblyConfig

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A document segment with formatting metadata"""
    content: str
    start_pos: int
    end_pos: int
    formatting: Dict[str, Any]
    segment_type: str = 'text'
    
    @property
    def length(self) -> int:
        return len(self.content)


class FormattingPreservingPartitionManager:
    """
    Partition manager that preserves original formatting during segmentation.
    """
    
    def __init__(self, config: DisassemblyConfig = None):
        """
        Initialize with configuration.
        
        Args:
            config: DisassemblyConfig instance
        """
        self.config = config or DisassemblyConfig()
        
    def create_partitions(self, text_or_windows: Union[str, List[str], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Create partitions compatible with existing code.
        
        Args:
            text_or_windows: Either a single text string, list of text strings, or list of window dicts
            
        Returns:
            List of partition dictionaries
        """
        # Handle different input types
        if isinstance(text_or_windows, str):
            # Single text string
            text = text_or_windows
        elif isinstance(text_or_windows, list):
            if not text_or_windows:
                return []
            
            # Check if it's a list of strings or list of dicts
            if isinstance(text_or_windows[0], str):
                # List of text strings - join them
                text = '\n\n'.join(text_or_windows)
            elif isinstance(text_or_windows[0], dict):
                # List of window dictionaries - extract content
                text = '\n\n'.join(w.get('content', '') for w in text_or_windows)
            else:
                # Unknown type
                raise ValueError(f"Unexpected type in list: {type(text_or_windows[0])}")
        else:
            raise ValueError(f"Expected str or list, got {type(text_or_windows)}")
        
        segments = self.partition_text(text)
        
        # Convert segments to expected format
        partitions = []
        for i, segment in enumerate(segments):
            partition = {
                'id': i,
                'content': segment.content,
                'start_pos': segment.start_pos,
                'end_pos': segment.end_pos,
                'segment_type': segment.segment_type,
                'formatting': segment.formatting,
                'metadata': {
                    'length': segment.length,
                    'has_overlap': i > 0 and segment.start_pos < segments[i-1].end_pos if i > 0 else False
                }
            }
            partitions.append(partition)
        
        return partitions
    
    def partition_text(self, text: str) -> List[Segment]:
        """
        Partition text while preserving all formatting.
        
        Args:
            text: Input text to partition
            
        Returns:
            List of Segment objects with formatting preserved
        """
        # First, identify special sections
        special_sections = self._identify_special_sections(text)
        
        # Create segments respecting boundaries and formatting
        segments = []
        current_pos = 0
        
        for section in special_sections:
            # Add any text before this section
            if section['start'] > current_pos:
                text_before = text[current_pos:section['start']]
                if text_before.strip():
                    segments.extend(self._segment_regular_text(
                        text_before, 
                        current_pos
                    ))
            
            # Add the special section
            segments.append(self._create_special_segment(
                text[section['start']:section['end']],
                section['start'],
                section['type'],
                section.get('metadata', {})
            ))
            
            current_pos = section['end']
        
        # Add any remaining text
        if current_pos < len(text):
            remaining_text = text[current_pos:]
            if remaining_text.strip():
                segments.extend(self._segment_regular_text(
                    remaining_text,
                    current_pos
                ))
        
        # Apply overlap for percolation
        if self.config.overlap_ratio > 0:
            segments = self._apply_overlap(segments, text)
        
        return segments
    
    def _identify_special_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify special sections that need intact preservation"""
        sections = []
        
        # Code blocks
        if self.config.preserve_code_blocks:
            sections.extend(self._find_code_blocks(text))
        
        # Conversations
        if self.config.detect_conversations:
            sections.extend(self._find_conversation_sections(text))
        
        # Lists
        if self.config.detect_lists:
            sections.extend(self._find_list_sections(text))
        
        # Sort by start position and merge overlapping
        sections.sort(key=lambda x: x['start'])
        return self._merge_overlapping_sections(sections)
    
    def _find_code_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Find code blocks in text"""
        sections = []
        
        # Markdown code blocks
        for match in re.finditer(r'```[\s\S]*?```', text):
            sections.append({
                'start': match.start(),
                'end': match.end(),
                'type': 'code_block',
                'metadata': {
                    'language': self._extract_code_language(match.group(0)),
                    'indentation': self._get_indentation_at(text, match.start())
                }
            })
        
        # Indented code blocks (4 spaces or tab)
        for match in re.finditer(r'(?:^|\n)((?:[ ]{4,}|\t).*(?:\n(?:[ ]{4,}|\t).*)*)', text, re.MULTILINE):
            sections.append({
                'start': match.start(1),
                'end': match.end(1),
                'type': 'indented_code',
                'metadata': {
                    'indentation': self._get_indentation_at(text, match.start(1))
                }
            })
        
        return sections
    
    def _find_conversation_sections(self, text: str) -> List[Dict[str, Any]]:
        """Find conversation sections"""
        sections = []
        
        # Pattern for speaker turns
        pattern = r'(?:^|\n)((?:Speaker\s+)?[A-Za-z0-9]+):\s*(.+?)(?=\n(?:Speaker\s+)?[A-Za-z0-9]+:|$)'
        
        matches = list(re.finditer(pattern, text, re.MULTILINE | re.DOTALL))
        
        if matches:
            # Group consecutive speaker turns
            start = matches[0].start()
            end = matches[-1].end()
            
            sections.append({
                'start': start,
                'end': end,
                'type': 'conversation',
                'metadata': {
                    'speakers': list(set(m.group(1) for m in matches)),
                    'turn_count': len(matches)
                }
            })
        
        return sections
    
    def _find_list_sections(self, text: str) -> List[Dict[str, Any]]:
        """Find list sections"""
        sections = []
        
        # Bullet lists
        bullet_pattern = r'(?:^|\n)((?:[-*+]\s+.+\n?)+)'
        for match in re.finditer(bullet_pattern, text, re.MULTILINE):
            sections.append({
                'start': match.start(1),
                'end': match.end(1),
                'type': 'bullet_list',
                'metadata': {
                    'indentation': self._get_indentation_at(text, match.start(1))
                }
            })
        
        # Numbered lists
        number_pattern = r'(?:^|\n)((?:\d+\.\s+.+\n?)+)'
        for match in re.finditer(number_pattern, text, re.MULTILINE):
            sections.append({
                'start': match.start(1),
                'end': match.end(1),
                'type': 'numbered_list',
                'metadata': {
                    'indentation': self._get_indentation_at(text, match.start(1))
                }
            })
        
        return sections
    
    def _segment_regular_text(self, text: str, base_pos: int) -> List[Segment]:
        """Segment regular text while preserving formatting"""
        segments = []
        
        # Split by paragraphs if configured
        if self.config.respect_paragraph_boundaries:
            paragraphs = re.split(r'\n\s*\n', text)
            current_pos = 0
            
            for para in paragraphs:
                if not para.strip():
                    current_pos += len(para) + 2  # Account for double newline
                    continue
                
                # Check if paragraph needs further segmentation
                if len(para) > self.config.max_segment_length:
                    # Split by sentences
                    segments.extend(self._segment_by_sentences(
                        para,
                        base_pos + current_pos
                    ))
                else:
                    segments.append(Segment(
                        content=para,
                        start_pos=base_pos + current_pos,
                        end_pos=base_pos + current_pos + len(para),
                        formatting=self._extract_formatting(para),
                        segment_type='paragraph'
                    ))
                
                current_pos += len(para) + 2
        else:
            # Direct sentence segmentation
            segments.extend(self._segment_by_sentences(text, base_pos))
        
        return segments
    
    def _segment_by_sentences(self, text: str, base_pos: int) -> List[Segment]:
        """Segment text by sentences"""
        segments = []
        
        # Simple sentence splitter (can be improved)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_pos = 0
        current_segment = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence exceeds target length
            if current_length + sentence_length > self.config.target_segment_length and current_segment:
                # Create segment from accumulated sentences
                segment_text = ' '.join(current_segment)
                segments.append(Segment(
                    content=segment_text,
                    start_pos=base_pos + current_pos - current_length,
                    end_pos=base_pos + current_pos,
                    formatting=self._extract_formatting(segment_text),
                    segment_type='text'
                ))
                
                current_segment = [sentence]
                current_length = sentence_length
            else:
                current_segment.append(sentence)
                current_length += sentence_length + 1  # +1 for space
            
            current_pos += sentence_length + 1
        
        # Add remaining sentences
        if current_segment:
            segment_text = ' '.join(current_segment)
            segments.append(Segment(
                content=segment_text,
                start_pos=base_pos + current_pos - current_length,
                end_pos=base_pos + current_pos,
                formatting=self._extract_formatting(segment_text),
                segment_type='text'
            ))
        
        return segments
    
    def _create_special_segment(self, content: str, start_pos: int, 
                               segment_type: str, metadata: Dict[str, Any]) -> Segment:
        """Create a special segment with metadata"""
        formatting = self._extract_formatting(content)
        formatting.update(metadata)
        
        return Segment(
            content=content,
            start_pos=start_pos,
            end_pos=start_pos + len(content),
            formatting=formatting,
            segment_type=segment_type
        )
    
    def _extract_formatting(self, text: str) -> Dict[str, Any]:
        """Extract formatting information from text"""
        formatting = {}
        
        # Indentation (part of whitespace preservation)
        if self.config.preserve_whitespace:
            indent_match = re.match(r'^(\s*)', text)
            if indent_match:
                formatting['indentation'] = indent_match.group(1)
        
        # Line breaks
        if self.config.preserve_line_breaks:
            formatting['line_breaks'] = text.count('\n')
            formatting['ends_with_newline'] = text.endswith('\n')
        
        # Whitespace
        if self.config.preserve_whitespace:
            formatting['leading_space'] = len(text) - len(text.lstrip())
            formatting['trailing_space'] = len(text) - len(text.rstrip())
        
        # Markdown formatting
        if self.config.preserve_markdown:
            formatting['has_bold'] = bool(re.search(r'\*\*[^*]+\*\*', text))
            formatting['has_italic'] = bool(re.search(r'\*[^*]+\*', text))
            formatting['has_code'] = bool(re.search(r'`[^`]+`', text))
            formatting['heading_level'] = self._get_heading_level(text)
        
        return formatting
    
    def _get_heading_level(self, text: str) -> Optional[int]:
        """Get markdown heading level if text is a heading"""
        match = re.match(r'^(#{1,6})\s+', text)
        if match:
            return len(match.group(1))
        return None
    
    def _get_indentation_at(self, text: str, pos: int) -> str:
        """Get indentation at a specific position"""
        # Find start of line
        line_start = text.rfind('\n', 0, pos) + 1
        indent_match = re.match(r'^(\s*)', text[line_start:])
        return indent_match.group(1) if indent_match else ''
    
    def _extract_code_language(self, code_block: str) -> Optional[str]:
        """Extract language from markdown code block"""
        match = re.match(r'```(\w+)', code_block)
        return match.group(1) if match else None
    
    def _merge_overlapping_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge overlapping sections"""
        if not sections:
            return []
        
        merged = [sections[0]]
        
        for section in sections[1:]:
            last = merged[-1]
            if section['start'] <= last['end']:
                # Merge sections
                last['end'] = max(last['end'], section['end'])
                # Combine metadata
                if 'metadata' in section:
                    last.setdefault('metadata', {}).update(section['metadata'])
            else:
                merged.append(section)
        
        return merged
    
    def _apply_overlap(self, segments: List[Segment], original_text: str) -> List[Segment]:
        """Apply overlap between segments for percolation"""
        if len(segments) <= 1:
            return segments
        
        overlapped_segments = []
        
        for i, segment in enumerate(segments):
            if i == 0:
                # First segment - add overlap at end
                overlap_length = int(segment.length * self.config.overlap_ratio)
                if i + 1 < len(segments):
                    next_segment = segments[i + 1]
                    overlap_content = original_text[segment.start_pos:next_segment.start_pos + overlap_length]
                    
                    overlapped_segments.append(Segment(
                        content=overlap_content,
                        start_pos=segment.start_pos,
                        end_pos=segment.end_pos + overlap_length,
                        formatting=segment.formatting,
                        segment_type=segment.segment_type
                    ))
            elif i == len(segments) - 1:
                # Last segment - add overlap at beginning
                overlap_length = int(segment.length * self.config.overlap_ratio)
                prev_segment = segments[i - 1]
                overlap_start = max(prev_segment.end_pos - overlap_length, prev_segment.start_pos)
                overlap_content = original_text[overlap_start:segment.end_pos]
                
                overlapped_segments.append(Segment(
                    content=overlap_content,
                    start_pos=overlap_start,
                    end_pos=segment.end_pos,
                    formatting=segment.formatting,
                    segment_type=segment.segment_type
                ))
            else:
                # Middle segment - add overlap at both ends
                overlap_length = int(segment.length * self.config.overlap_ratio / 2)
                prev_segment = segments[i - 1]
                next_segment = segments[i + 1]
                
                overlap_start = max(prev_segment.end_pos - overlap_length, prev_segment.start_pos)
                overlap_end = min(next_segment.start_pos + overlap_length, next_segment.end_pos)
                overlap_content = original_text[overlap_start:overlap_end]
                
                overlapped_segments.append(Segment(
                    content=overlap_content,
                    start_pos=overlap_start,
                    end_pos=overlap_end,
                    formatting=segment.formatting,
                    segment_type=segment.segment_type
                ))
        
        return overlapped_segments
    
    def reconstruct_formatting(self, segments: List[Segment]) -> str:
        """
        Reconstruct document with original formatting preserved.
        
        Args:
            segments: List of segments to reconstruct
            
        Returns:
            Reconstructed text with formatting preserved
        """
        if not segments:
            return ""
        
        # Sort segments by position
        sorted_segments = sorted(segments, key=lambda s: s.start_pos)
        
        parts = []
        
        for i, segment in enumerate(sorted_segments):
            content = segment.content
            formatting = segment.formatting
            
            # Apply formatting
            if formatting.get('indentation'):
                # Ensure proper indentation
                lines = content.split('\n')
                indented_lines = []
                for j, line in enumerate(lines):
                    if j == 0 and not line.startswith(formatting['indentation']):
                        line = formatting['indentation'] + line
                    indented_lines.append(line)
                content = '\n'.join(indented_lines)
            
            # Preserve line breaks
            if i > 0 and segment.segment_type == 'paragraph':
                # Add paragraph spacing
                parts.append('\n\n')
            elif i > 0 and not parts[-1].endswith('\n'):
                parts.append('\n')
            
            parts.append(content)
        
        return ''.join(parts)