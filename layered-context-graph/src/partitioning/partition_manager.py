class PartitionManager:
    def __init__(self, overlap_ratio=0.25, target_segment_length=400, max_rounds=5):
        self.overlap_ratio = overlap_ratio
        self.target_segment_length = target_segment_length  # Target average segment length
        self.max_rounds = max_rounds  # Maximum segmentation rounds
        self.partitions = []
        self.segmentation_history = []  # Track each round of segmentation
        self.disassembly_rules = {
            'semantic_boundaries': True,
            'attention_clusters': True, 
            'percolation_thresholds': True,
            'instruction_markers': True
        }
        self.partition_metadata = []

    def create_partitions(self, context_windows):
        """ITERATIVE DISASSEMBLY: Break down text through multiple rounds until optimal segment size"""
        if context_windows is None:
            return []
        
        if isinstance(context_windows, str):
            # Start with initial basic split
            current_segments = self._initial_split(context_windows)
        elif isinstance(context_windows, list):
            # If it's already a list, process each window
            current_segments = []
            for window in context_windows:
                sub_partitions = self._initial_split(window)
                current_segments.extend(sub_partitions)
        else:
            current_segments = [str(context_windows)]
        
        # Apply iterative segmentation until optimal length
        final_segments = self._iterative_segmentation(current_segments)
        
        self.partitions = final_segments
        return final_segments
    
    def _iterative_segmentation(self, initial_segments):
        """Apply multiple rounds of segmentation until reaching target average length"""
        current_segments = initial_segments
        
        print(f"🔄 Starting iterative segmentation with {len(current_segments)} initial segments")
        
        for round_num in range(self.max_rounds):
            # Calculate current average segment length
            avg_length = self._calculate_average_length(current_segments)
            
            print(f"   Round {round_num + 1}: {len(current_segments)} segments, avg length: {avg_length:.1f} chars")
            
            # Check if we've reached target average length (within tolerance)
            tolerance = 50  # Within 50 characters of target
            if abs(avg_length - self.target_segment_length) <= tolerance:
                print(f"   ✅ Reached target segment length ({self.target_segment_length}) after {round_num + 1} rounds")
                break
            
            # Apply segmentation round based on current needs
            new_segments = self._apply_segmentation_round(current_segments, round_num, avg_length)
            
            # Record this round's results
            self.segmentation_history.append({
                'round': round_num + 1,
                'segments_before': len(current_segments),
                'segments_after': len(new_segments),
                'avg_length_before': avg_length,
                'avg_length_after': self._calculate_average_length(new_segments),
                'criteria': self._get_round_criteria(round_num),
                'action': 'split' if avg_length > self.target_segment_length else 'merge'
            })
            
            # Check for convergence (no significant change)
            if len(new_segments) == len(current_segments):
                print(f"   🔄 Converged after {round_num + 1} rounds (no change in segment count)")
                break
                
            current_segments = new_segments
        
        print(f"   📊 Final result: {len(current_segments)} segments, avg length: {self._calculate_average_length(current_segments):.1f} chars")
        return current_segments
    
    def _calculate_average_length(self, segments):
        """Calculate average character length of segments"""
        if not segments:
            return 0
        return sum(len(segment) for segment in segments) / len(segments)
    
    def _apply_segmentation_round(self, segments, round_num, current_avg_length):
        """Apply different segmentation criteria for each round"""
        
        if current_avg_length > self.target_segment_length:
            # Segments are too long - split them
            return self._split_segments(segments, round_num)
        else:
            # Segments are too short - merge them
            return self._merge_segments(segments, round_num)
    
    def _split_segments(self, segments, round_num):
        """Split segments that are too long using different criteria per round"""
        new_segments = []
        
        for segment in segments:
            if len(segment) > self.target_segment_length * 1.5:  # Split if 1.5x target
                split_segments = self._split_by_round_criteria(segment, round_num)
                new_segments.extend(split_segments)
            else:
                new_segments.append(segment)
        
        return new_segments
    
    def _merge_segments(self, segments, round_num):
        """Merge segments that are too short"""
        if not segments:
            return segments
            
        new_segments = []
        current_merged = ""
        
        for segment in segments:
            # Try to merge with current segment
            combined_length = len(current_merged) + len(segment)
            
            if combined_length <= self.target_segment_length * 1.2:  # Allow 20% over target
                current_merged = (current_merged + " " + segment).strip()
            else:
                # Add current merged segment and start new one
                if current_merged:
                    new_segments.append(current_merged)
                current_merged = segment
        
        # Add final merged segment
        if current_merged:
            new_segments.append(current_merged)
        
        return new_segments
    
    def _split_by_round_criteria(self, segment, round_num):
        """Split segment using different criteria based on round number"""
        
        if round_num == 0:
            # Round 1: Split by semantic boundaries (paragraphs, sentences)
            return self._split_by_semantic_boundaries(segment)
        elif round_num == 1:
            # Round 2: Split by attention patterns (if available) or syntactic boundaries
            return self._split_by_syntactic_boundaries(segment)
        elif round_num == 2:
            # Round 3: Split by instruction markers
            return self._split_by_instruction_markers(segment)
        else:
            # Round 4+: Simple character-based splitting as fallback
            return self._split_by_character_count(segment)
    
    def _split_by_semantic_boundaries(self, segment):
        """Split by paragraph and sentence boundaries"""
        # First try paragraph breaks
        paragraphs = segment.split('\n\n')
        if len(paragraphs) > 1:
            return [p.strip() for p in paragraphs if p.strip()]
        
        # Then try sentence breaks
        sentences = segment.split('. ')
        if len(sentences) > 1:
            # Group sentences to approach target length
            groups = []
            current_group = ""
            
            for sentence in sentences:
                if len(current_group) + len(sentence) < self.target_segment_length:
                    current_group = (current_group + ". " + sentence).strip()
                else:
                    if current_group:
                        groups.append(current_group)
                    current_group = sentence
            
            if current_group:
                groups.append(current_group)
                
            return groups
        
        return [segment]  # Can't split further
    
    def _split_by_syntactic_boundaries(self, segment):
        """Split by syntactic boundaries (commas, conjunctions, etc.)"""
        # Split by major punctuation
        split_chars = ['; ', ', and ', ', or ', ', but ', '. However, ', '. Therefore, ']
        
        for split_char in split_chars:
            if split_char in segment:
                parts = segment.split(split_char)
                if len(parts) > 1:
                    return [part.strip() for part in parts if part.strip()]
        
        return [segment]  # Can't split further
    
    def _split_by_instruction_markers(self, segment):
        """Split by instruction markers"""
        markers = ['<MATH>', '<DIALOGUE>', '<MEMORY>', '<TRACK>', 
                  '<QWQ_REASONING>', '<QWQ_CONCLUSION>', '<QWQ_EXAMPLE>']
        
        for marker in markers:
            if marker in segment:
                parts = segment.split(marker)
                result = []
                for i, part in enumerate(parts):
                    if i > 0:  # Add marker back to parts after the first
                        part = marker + part
                    if part.strip():
                        result.append(part.strip())
                if len(result) > 1:
                    return result
        
        return [segment]  # Can't split further
    
    def _split_by_character_count(self, segment):
        """Fallback: split by character count while preserving word boundaries"""
        if len(segment) <= self.target_segment_length:
            return [segment]
        
        # Split at word boundaries near target length
        words = segment.split()
        parts = []
        current_part = ""
        
        for word in words:
            if len(current_part) + len(word) + 1 <= self.target_segment_length:
                current_part = (current_part + " " + word).strip()
            else:
                if current_part:
                    parts.append(current_part)
                current_part = word
        
        if current_part:
            parts.append(current_part)
        
        return parts
    
    def _get_round_criteria(self, round_num):
        """Get description of criteria used in this round"""
        criteria_map = {
            0: "semantic_boundaries",
            1: "syntactic_boundaries", 
            2: "instruction_markers",
            3: "character_count"
        }
        return criteria_map.get(round_num, "character_count")
    
    def get_segmentation_summary(self):
        """Get summary of the segmentation process"""
        return {
            'total_rounds': len(self.segmentation_history),
            'final_segment_count': len(self.partitions),
            'target_length': self.target_segment_length,
            'final_avg_length': self._calculate_average_length(self.partitions),
            'rounds_detail': self.segmentation_history
        }

    def manage_partitions(self):
        """Manage the created partitions"""
        # Return current partitions for now
        return self.partitions
    
    def _initial_split(self, text):
        """Perform initial text splitting based on natural boundaries"""
        if len(text) <= self.target_segment_length:
            return [text]
        
        # Try semantic splitting first
        semantic_splits = self._split_by_semantic_boundaries(text)
        if len(semantic_splits) > 1:
            return semantic_splits
        
        # Fall back to syntactic boundaries
        syntactic_splits = self._split_by_syntactic_boundaries(text)
        if len(syntactic_splits) > 1:
            return syntactic_splits
        
        # Final fallback: character-based splitting
        return self._split_by_character_count(text)