import re
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from models.baai_model import BAAIModel

logger = logging.getLogger(__name__)

@dataclass
class EnrichedSegment:
    """A structured data class for a document segment."""
    id: str
    content: str
    start_pos: int
    end_pos: int
    has_math: bool = False
    has_code: bool = False
    tag: str = 'track'
    metadata: Dict[str, Any] = field(default_factory=dict)

class PartitionManager:
    """
    A definitive, model-driven partition manager that uses semantic similarity
    to implement K-rule-based segmentation and produces enriched segment objects.
    """

    def __init__(self, embedding_model: BAAIModel, min_segment_len: int = 100, cohesion_threshold: float = 0.5):
        self.embedding_model = embedding_model
        self.min_segment_len = min_segment_len
        self.cohesion_threshold = cohesion_threshold

    def create_partitions(self, text: str) -> List[EnrichedSegment]:
        """
        Creates enriched partitions by applying a series of K-rules.
        """
        # K1: Initial Disassembly - Isolate special content
        special_segments, regular_text_parts = self._extract_special_content(text)

        # Process regular text parts
        regular_segments = []
        for part in regular_text_parts:
            # K1 cont'd: Find semantic boundaries in regular text
            semantic_segments = self._segment_by_semantic_cohesion(part['content'], part['start'])
            regular_segments.extend(semantic_segments)

        # Combine and sort all segments
        all_initial_segments = sorted(special_segments + regular_segments, key=lambda x: x['start'])
        
        # K2/K3: Refine into sentences
        refined_segments = self._refine_to_sentences(all_initial_segments)

        # K4: Merge short segments
        merged_segments = self._merge_short_segments(refined_segments)

        # Create final EnrichedSegment objects
        final_enriched_segments = []
        for i, seg_dict in enumerate(merged_segments):
            final_enriched_segments.append(
                self._create_enriched_segment(f"seg_{i}", seg_dict['content'], seg_dict['start'])
            )

        logger.info(f"Partitioning complete. Produced {len(final_enriched_segments)} final segments.")
        return final_enriched_segments

    def _create_enriched_segment(self, id_str: str, content: str, start: int) -> EnrichedSegment:
        """Creates a single EnrichedSegment with metadata."""
        return EnrichedSegment(
            id=id_str,
            content=content,
            start_pos=start,
            end_pos=start + len(content),
            has_math=bool(re.search(r'\\\[.*\\\]|\\\(.*\\\)|[$]{1,2}[^$]+[$]{1,2}', content)),
            has_code='```' in content,
            tag='track'
        )

    def _extract_special_content(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """(K1) Extracts code blocks, returning them and the remaining text parts."""
        code_block_pattern = re.compile(r'(```[\s\S]*?```)')
        
        special_segments = []
        regular_text_parts = []
        last_end = 0

        for match in code_block_pattern.finditer(text):
            if match.start() > last_end:
                regular_text_parts.append({'content': text[last_end:match.start()], 'start': last_end})
            
            special_segments.append({'content': match.group(0), 'start': match.start()})
            last_end = match.end()

        if last_end < len(text):
            regular_text_parts.append({'content': text[last_end:], 'start': last_end})
            
        return special_segments, regular_text_parts

    def _segment_by_semantic_cohesion(self, text: str, base_offset: int) -> List[Dict]:
        """(K1) Uses embedding similarity to find semantic boundaries."""
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sentences) <= 1:
            return [{'content': text, 'start': base_offset}] if text else []

        embeddings = self.embedding_model.encode(sentences)
        if embeddings.ndim == 1: embeddings = np.expand_dims(embeddings, axis=0)
            
        similarities = [np.dot(embeddings[i], embeddings[i+1]) for i in range(len(embeddings) - 1)]
        
        segments = []
        current_segment_sentences = []
        current_pos_in_text = 0

        for i, sentence in enumerate(sentences):
            current_segment_sentences.append(sentence)
            if i < len(similarities) and similarities[i] < self.cohesion_threshold:
                segment_text = " ".join(current_segment_sentences)
                start_pos = text.find(segment_text, current_pos_in_text)
                segments.append({'content': segment_text, 'start': start_pos + base_offset})
                current_pos_in_text = start_pos + len(segment_text)
                current_segment_sentences = []
        
        if current_segment_sentences:
            segment_text = " ".join(current_segment_sentences)
            start_pos = text.find(segment_text, current_pos_in_text)
            segments.append({'content': segment_text, 'start': start_pos + base_offset})

        return segments

    def _refine_to_sentences(self, segments: List[Dict]) -> List[Dict]:
        """(K2/K3) Refines larger segments into sentences."""
        refined = []
        for seg in segments:
            content = seg['content']
            start_offset = seg['start']
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if s.strip()]
            current_pos_in_seg = 0
            for s in sentences:
                start_pos = content.find(s, current_pos_in_seg)
                if start_pos != -1:
                    refined.append({'content': s, 'start': start_offset + start_pos})
                    current_pos_in_seg = start_pos + len(s)
        return refined

    def _merge_short_segments(self, segments: List[Dict]) -> List[Dict]:
        """(K4) Merges segments that are shorter than the minimum length."""
        if not segments: return []

        merged = []
        current_seg_dict = segments[0]
        
        for i in range(1, len(segments)):
            next_seg_dict = segments[i]
            if len(current_seg_dict['content']) < self.min_segment_len:
                similarity = self.embedding_model.compute_similarity(current_seg_dict['content'], next_seg_dict['content'])
                if similarity > 0.75:
                    current_seg_dict['content'] += " " + next_seg_dict['content']
                else:
                    merged.append(current_seg_dict)
                    current_seg_dict = next_seg_dict
            else:
                merged.append(current_seg_dict)
                current_seg_dict = next_seg_dict
        
        if current_seg_dict:
            merged.append(current_seg_dict)
            
        return merged
