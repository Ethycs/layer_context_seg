class PartitionManager:
    def __init__(self, overlap_ratio=0.25, target_segment_length=400, max_rounds=5):
        self.overlap_ratio = overlap_ratio
        self.target_segment_length = target_segment_length  # Target average segment length
        self.max_rounds = max_rounds  # Maximum segmentation rounds
        self.partitions = []
        self.segmentation_history = []  # Track each round of segmentation
        self.disassembly_rules = {
            'semantic_boundaries': True,
            'attention_clusters': False,  # Disable aggressive sentence splitting
            'percolation_thresholds': True,
            'instruction_markers': True,
            'conversation_boundaries': False  # New: for conversation tracking
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
                # For larger documents, increase target segment length
                if len(window) > 5000:
                    self.target_segment_length = 1500
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
            tolerance = 200  # Increased tolerance for more flexibility
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
            if len(segment) > self.target_segment_length * 2:  # Only split if significantly over target
                split_segments = self._split_by_round_criteria(segment, round_num)
                new_segments.extend(split_segments)
            else:
                new_segments.append(segment)
        
        return new_segments
    
    def _merge_segments(self, segments, round_num):
        """Merge segments that are too short - be more aggressive about merging"""
        if not segments:
            return segments
            
        new_segments = []
        current_merged = ""
        
        for segment in segments:
            # Try to merge with current segment
            combined_length = len(current_merged) + len(segment)
            
            # Be more aggressive - allow up to 2x target for merging
            if combined_length <= self.target_segment_length * 2:
                if current_merged:
                    current_merged = current_merged + " " + segment
                else:
                    current_merged = segment
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
            # Round 1: Split by semantic boundaries (paragraphs)
            return self._split_by_semantic_boundaries(segment)
        elif round_num == 1:
            # Round 2: Split by larger syntactic units (multiple sentences)
            return self._split_by_paragraph_boundaries(segment)
        elif round_num == 2:
            # Round 3: Split by instruction markers
            return self._split_by_instruction_markers(segment)
        else:
            # Round 4+: Character-based splitting
            return self._split_by_character_count(segment)
    
    def _apply_disassembly_rules(self, text):
        """
        Rule K1: Initial Disassembly Rules
        Apply semantic, attention, percolation, and instruction marker boundaries
        """
        segments = [text]
        
        # Apply each disassembly rule if enabled
        if self.disassembly_rules.get('semantic_boundaries', True):
            segments = self._split_by_semantic_boundaries(segments)
            
        if self.disassembly_rules.get('attention_clusters', True):
            segments = self._split_by_attention_clusters(segments)
            
        if self.disassembly_rules.get('percolation_thresholds', True):
            segments = self._split_by_percolation_thresholds(segments)
            
        if self.disassembly_rules.get('instruction_markers', True):
            segments = self._split_by_instruction_markers(segments)
            
        if self.disassembly_rules.get('conversation_boundaries', False):
            segments = self._apply_conversation_disassembly_rules(segments)
            
        return segments
    
    def _split_by_semantic_boundaries(self, segments):
        """Split segments at semantic boundaries (paragraphs, topic shifts)"""
        if isinstance(segments, str):
            segments = [segments]
            
        new_segments = []
        for segment in segments:
            # Split by double newlines (paragraphs)
            paragraphs = segment.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip() and len(paragraph.strip()) > 50:  # Minimum paragraph size
                    new_segments.append(paragraph.strip())
        return new_segments
    
    def _split_by_paragraph_boundaries(self, segment):
        """Split by paragraph-like boundaries (multiple sentences together)"""
        if isinstance(segment, str):
            # Split by double newlines or multiple spaces
            import re
            # Look for paragraph breaks or topic shifts
            parts = re.split(r'\n\n|\n(?=[A-Z*#])', segment)
            
            result = []
            for part in parts:
                part = part.strip()
                if len(part) > 100:  # Minimum size for a meaningful chunk
                    result.append(part)
                elif result and len(result[-1]) + len(part) < self.target_segment_length * 1.5:
                    # Merge small parts with previous
                    result[-1] += " " + part
                else:
                    result.append(part)
            
            return [r for r in result if len(r.strip()) > 50]
        else:
            return [segment]
    
    def _split_by_attention_clusters(self, segments):
        """Split segments using attention pattern analysis - less aggressive"""
        # Only split very long segments
        new_segments = []
        for segment in segments:
            if len(segment) > self.target_segment_length * 3:
                # Split into roughly equal parts
                mid = len(segment) // 2
                new_segments.extend([segment[:mid], segment[mid:]])
            else:
                new_segments.append(segment)
        return new_segments
    
    def _split_by_percolation_thresholds(self, segments):
        """Apply percolation theory boundaries"""
        # Use overlap ratio to determine optimal split points
        new_segments = []
        for segment in segments:
            words = segment.split()
            # Only split if segment is very large
            if len(words) > self.target_segment_length // 5:
                # Split at natural percolation boundaries
                mid_point = len(words) // 2
                overlap_size = int(len(words) * self.overlap_ratio)
                
                part1 = ' '.join(words[:mid_point + overlap_size])
                part2 = ' '.join(words[mid_point:])
                
                if len(part1) > 200 and len(part2) > 200:  # Ensure meaningful size
                    new_segments.extend([part1, part2])
                else:
                    new_segments.append(segment)
            else:
                new_segments.append(segment)
        return new_segments
    
    def _split_by_instruction_markers(self, segments):
        """Split at special instruction markers"""
        if isinstance(segments, str):
            segments = [segments]
            
        new_segments = []
        for segment in segments:
            # Look for instruction markers like <SEGMENT>, <RELATE>, etc.
            import re
            markers = re.findall(r'<[A-Z_]+>.*?</[A-Z_]+>', segment)
            if markers:
                # Split around markers
                parts = re.split(r'<[A-Z_]+>.*?</[A-Z_]+>', segment)
                for i, part in enumerate(parts):
                    if part.strip() and len(part.strip()) > 50:
                        new_segments.append(part.strip())
                    if i < len(markers):
                        new_segments.append(markers[i])
            else:
                new_segments.append(segment)
        return new_segments
    
    def _split_by_syntactic_boundaries(self, segment):
        """Split by syntactic boundaries - but keep larger chunks"""
        if isinstance(segment, str):
            # Don't split by single sentences - look for multiple sentence groups
            import re
            # Split on multiple punctuation or clear topic breaks
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z*#])', segment)
            
            # Merge small parts to create larger chunks
            result = []
            current = ""
            for part in parts:
                if len(current) + len(part) < self.target_segment_length:
                    current = (current + " " + part).strip()
                else:
                    if current:
                        result.append(current)
                    current = part
            
            if current:
                result.append(current)
            
            return [r for r in result if len(r) > 100]
        else:
            # If it's a list, process each item
            result = []
            for item in segment:
                result.extend(self._split_by_syntactic_boundaries(item))
            return result
    
    def _split_by_character_count(self, segment):
        """Simple character-based splitting as fallback"""
        if isinstance(segment, str):
            if len(segment) <= self.target_segment_length:
                return [segment]
            
            # Split into chunks of target length, but try to break at word boundaries
            chunks = []
            words = segment.split()
            current_chunk = []
            current_length = 0
            
            for word in words:
                word_length = len(word) + 1  # +1 for space
                if current_length + word_length > self.target_segment_length and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = word_length
                else:
                    current_chunk.append(word)
                    current_length += word_length
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            return chunks
        else:
            # If it's a list, process each item
            result = []
            for item in segment:
                result.extend(self._split_by_character_count(item))
            return result
    
    def _get_round_criteria(self, round_num):
        """Get the criteria used for a specific round"""
        criteria_map = {
            0: "semantic_boundaries",
            1: "paragraph_boundaries", 
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
        if len(text) <= self.target_segment_length * 2:
            return [text]
        
        # Try paragraph boundaries first
        paragraph_splits = self._split_by_paragraph_boundaries(text)
        if len(paragraph_splits) > 1 and all(len(p) > 200 for p in paragraph_splits):
            return paragraph_splits
        
        # Fall back to semantic boundaries
        semantic_splits = self._split_by_semantic_boundaries(text)
        if len(semantic_splits) > 1 and all(len(s) > 200 for s in semantic_splits):
            return semantic_splits
        
        # Final fallback: character-based splitting
        return self._split_by_character_count(text)
    
    def _apply_conversation_disassembly_rules(self, segments):
        """
        Apply conversation-specific semantic boundaries using attention patterns.
        This method leverages the LLM's attention to identify:
        - Speaker turn boundaries
        - Question-answer pairs  
        - Topic shifts and idea evolution
        - Reference patterns
        """
        import re
        
        if isinstance(segments, str):
            segments = [segments]
        
        new_segments = []
        conversation_metadata = []
        
        for segment in segments:
            # Basic speaker pattern detection for initial splitting
            speaker_pattern = r'(Speaker\s+[A-Za-z0-9]+:|^[A-Za-z0-9]+:|\n[A-Za-z0-9]+:)'
            
            # Split by speaker turns as initial boundaries
            parts = re.split(f'({speaker_pattern})', segment)
            
            current_content = ""
            current_speaker = None
            
            for i, part in enumerate(parts):
                if re.match(speaker_pattern, part):
                    # Save previous content if exists
                    if current_content.strip():
                        segment_data = {
                            'content': current_content.strip(),
                            'speaker': current_speaker,
                            'type': 'conversation_turn',
                            'requires_attention_analysis': True
                        }
                        new_segments.append(current_content.strip())
                        conversation_metadata.append(segment_data)
                    
                    # Update speaker
                    current_speaker = part.strip().rstrip(':')
                    current_content = part
                else:
                    current_content += part
            
            # Don't forget the last segment
            if current_content.strip():
                segment_data = {
                    'content': current_content.strip(),
                    'speaker': current_speaker,
                    'type': 'conversation_turn',
                    'requires_attention_analysis': True
                }
                new_segments.append(current_content.strip())
                conversation_metadata.append(segment_data)
        
        # Store metadata for attention-based analysis in graph construction
        if hasattr(self, 'partition_metadata'):
            self.partition_metadata.extend(conversation_metadata)
        else:
            self.partition_metadata = conversation_metadata
        
        return new_segments