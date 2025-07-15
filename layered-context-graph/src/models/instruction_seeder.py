"""
Instruction Seeder Module
------------------------
This module implements instruction seeding for attention guidance.
It inserts natural language instructions into text to guide the attention
heads toward specific behaviors like boundary detection or semantic clustering.
"""

import re
import random
from typing import Dict, List, Optional, Union, Tuple

class InstructionSeeder:
    """Class for seeding instructions in text to guide attention mechanisms"""
    
    def __init__(self, instruction_types: Optional[Dict[str, str]] = None):
        """
        Initialize the instruction seeder with predefined instruction types
        
        Args:
            instruction_types: Dictionary mapping instruction tags to descriptions
        """
        self.instruction_types = instruction_types or {
            # Original instruction types
            "ANALYZE": "Look for causal relationships",
            "CONNECT": "Link related concepts across time",
            "ABSTRACT": "Extract high-level themes",
            "BOUNDARY": "Detect topic or section changes",
            "CLASSIFY": "Group similar content together",
            "PRIORITIZE": "Focus on important information",
            "CODE": "Preserve code structure and semantics",
            "MATH": "Maintain mathematical relationships",
            "NARRATIVE": "Follow story or argument flow",
            "TECHNICAL": "Preserve technical details",
            
            # Conversation-specific instruction types
            "SPEAKER_BOUNDARY": "Identify speaker turn changes",
            "TOPIC_EVOLUTION": "Track how ideas develop through discussion",
            "REFERENCE": "Highlight back-references to earlier points",
            "AGREEMENT": "Mark consensus and agreement between speakers",
            "DISAGREEMENT": "Identify conflicts and contradictions",
            "QUESTION": "Detect questions being asked",
            "ANSWER": "Identify responses to questions",
            "CLARIFICATION": "Mark clarifying statements",
            "SUMMARY": "Identify summarizing statements",
            "CONCLUSION": "Detect concluding remarks"
        }
    
    def seed_instructions(self, text: str, density: float = 0.1) -> str:
        """
        Insert instruction markers throughout text to guide attention
        
        Args:
            text: The input text to process
            density: The approximate density of instructions (0.0-1.0)
                     Higher values insert more instructions
        
        Returns:
            Text with instruction markers inserted
        """
        if not text or not text.strip():
            return text
            
        # Split text into sentences or paragraphs for seeding
        chunks = self._split_into_seedable_chunks(text)
        
        # Calculate how many instructions to insert based on density
        num_chunks = len(chunks)
        num_instructions = max(1, int(num_chunks * density))
        
        # Randomly select chunks to seed (without replacement)
        if num_chunks > num_instructions:
            seed_indices = sorted(random.sample(range(num_chunks), num_instructions))
        else:
            seed_indices = list(range(num_chunks))
        
        # Insert instructions at the selected positions
        seeded_chunks = []
        for i, chunk in enumerate(chunks):
            if i in seed_indices:
                # Choose a random instruction type
                instr_type, instr_desc = random.choice(list(self.instruction_types.items()))
                
                # Insert the instruction before the chunk
                seeded_chunk = f"<{instr_type}>{chunk}</{instr_type}>"
                seeded_chunks.append(seeded_chunk)
            else:
                seeded_chunks.append(chunk)
        
        # Reassemble the text
        seeded_text = self._reassemble_chunks(seeded_chunks)
        
        # Add global processing instructions at the beginning
        global_instructions = self._create_global_instructions()
        result = f"{global_instructions}\n\n{seeded_text}"
        
        return result
    
    def seed_with_rules(self, text: str, rules: Dict[str, str]) -> str:
        """
        Seed text with specific rules for segmentation and reorganization
        
        Args:
            text: The input text
            rules: Dictionary with 'segmentation' and 'reorganization' rules
        
        Returns:
            Text with rule-based instruction markers
        """
        if not text or not isinstance(rules, dict):
            return text
            
        # Extract rules
        seg_rule = rules.get('segmentation', "Split at natural topic boundaries")
        reorg_rule = rules.get('reorganization', "Group by theme and importance")
        
        # Create rule-based prompt
        rule_prompt = f"""
        <processing_rules>
        <segmentation_rule>{seg_rule}</segmentation_rule>
        <reorganization_rule>{reorg_rule}</reorganization_rule>
        </processing_rules>
        
        """
        
        # Add some inline rule markers at key positions
        chunks = self._split_into_seedable_chunks(text)
        seeded_chunks = []
        
        # Seed at beginning, middle and near ends for better coverage
        key_positions = [0]
        if len(chunks) >= 3:
            key_positions.extend([len(chunks) // 2, len(chunks) - 1])
        
        for i, chunk in enumerate(chunks):
            if i in key_positions:
                if i == 0:
                    # Beginning: Focus on segmentation
                    seeded_chunks.append(f"<SEGMENT rule='{seg_rule}'>{chunk}</SEGMENT>")
                elif i == len(chunks) // 2:
                    # Middle: Focus on relationships
                    seeded_chunks.append(f"<CONNECT rule='{reorg_rule}'>{chunk}</CONNECT>")
                else:
                    # End: Focus on reorganization
                    seeded_chunks.append(f"<ORGANIZE rule='{reorg_rule}'>{chunk}</ORGANIZE>")
            else:
                seeded_chunks.append(chunk)
        
        # Reassemble with rule prompt at beginning
        seeded_text = self._reassemble_chunks(seeded_chunks)
        result = rule_prompt + seeded_text
        
        return result
    
    def _split_into_seedable_chunks(self, text: str) -> List[str]:
        """Split text into appropriate chunks for seeding instructions"""
        # Try to respect paragraph boundaries first
        if '\n\n' in text:
            chunks = text.split('\n\n')
            # Filter out empty chunks
            chunks = [c for c in chunks if c.strip()]
            if chunks:
                return chunks
        
        # Fall back to sentences if no paragraphs
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        # Group sentences into reasonably-sized chunks
        for sentence in sentences:
            if current_chunk and len(current_chunk + sentence) > 100:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def _reassemble_chunks(self, chunks: List[str]) -> str:
        """Reassemble chunks into cohesive text"""
        # Join with double newline to preserve paragraph structure
        return '\n\n'.join(chunks)
    
    def _create_global_instructions(self) -> str:
        """Create global processing instructions for the beginning of the text"""
        return """<processing_hints>
<ANALYZE>Look for causal relationships</ANALYZE>
<CONNECT>Link related concepts across time</CONNECT>
<ABSTRACT>Extract high-level themes</ABSTRACT>
</processing_hints>"""
    
    def program_with_natural_language(self, 
                                     segmentation_rule: str, 
                                     reorganization_rule: str) -> Dict[str, str]:
        """
        Program the system with plain English rules
        
        Args:
            segmentation_rule: Natural language rule for segmentation
            reorganization_rule: Natural language rule for reorganization
        
        Returns:
            Dictionary of encoded rules
        """
        return {
            'segmentation': segmentation_rule,
            'reorganization': reorganization_rule
        }
    
    def encode_rule_for_attention(self, text: str, rule: str) -> str:
        """
        Encode a natural language rule into the text for attention guidance
        
        Args:
            text: Input text to process
            rule: Natural language rule to apply
        
        Returns:
            Text with encoded rule
        """
        # Simple version - just prepend the rule
        return f"Rule: {rule}\n\nText: {text}"
    
    def seed_conversation_instructions(self, 
                                     text: str, 
                                     mode: str = 'timeline',
                                     density: float = 0.15) -> str:
        """
        Seed conversation-specific instructions based on the desired segmentation mode.
        
        Args:
            text: Conversation transcript
            mode: Segmentation mode ('timeline', 'speaker', 'evolution', 'topics')
            density: Instruction density
            
        Returns:
            Text with conversation-specific instructions
        """
        # Mode-specific instruction sets
        mode_instructions = {
            'timeline': ['SPEAKER_BOUNDARY', 'BOUNDARY', 'NARRATIVE'],
            'speaker': ['SPEAKER_BOUNDARY', 'QUESTION', 'ANSWER'],
            'evolution': ['TOPIC_EVOLUTION', 'REFERENCE', 'CLARIFICATION'],
            'topics': ['ABSTRACT', 'CLASSIFY', 'TOPIC_EVOLUTION'],
            'consensus': ['AGREEMENT', 'DISAGREEMENT', 'CONCLUSION']
        }
        
        # Get instructions for this mode
        selected_types = mode_instructions.get(mode, ['BOUNDARY', 'CONNECT'])
        
        # Create custom instruction set
        custom_instructions = {k: self.instruction_types[k] 
                             for k in selected_types 
                             if k in self.instruction_types}
        
        # Temporarily override instruction types
        original_types = self.instruction_types
        self.instruction_types = custom_instructions
        
        # Apply targeted seeding
        seeded_text = self.seed_instructions(text, density)
        
        # Add mode-specific global instruction
        mode_instruction = f"<SEGMENT_RULE>Organize by {mode}</SEGMENT_RULE>\n"
        seeded_text = mode_instruction + seeded_text
        
        # Restore original instruction types
        self.instruction_types = original_types
        
        return seeded_text
    
    def seed_speaker_boundaries(self, text: str) -> str:
        """
        Specifically seed speaker turn boundaries for better segmentation.
        
        Args:
            text: Conversation text
            
        Returns:
            Text with speaker boundary markers
        """
        import re
        
        # Pattern to match speaker labels
        speaker_pattern = r'((?:^|\n)(?:Speaker\s+)?[A-Za-z0-9]+):\s*'
        
        # Find all speaker turns
        parts = re.split(f'({speaker_pattern})', text)
        
        seeded_parts = []
        for i, part in enumerate(parts):
            if re.match(speaker_pattern, part):
                # This is a speaker label - add boundary instruction
                seeded_parts.append(f"<SPEAKER_BOUNDARY>{part}</SPEAKER_BOUNDARY>")
            else:
                seeded_parts.append(part)
        
        return ''.join(seeded_parts)
    
    def create_attention_bias_tensor(self, 
                                   instruction_positions: List[int],
                                   sequence_length: int,
                                   instruction_type: str = 'BOUNDARY') -> 'torch.Tensor':
        """
        Create a bias tensor to influence attention based on instruction positions.
        This tensor can be added to attention scores to guide the model.
        
        Args:
            instruction_positions: Positions where instructions were inserted
            sequence_length: Length of the sequence
            instruction_type: Type of instruction for determining bias strength
            
        Returns:
            Bias tensor of shape (sequence_length, sequence_length)
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for bias tensor creation")
        
        # Initialize bias matrix
        bias = torch.zeros((sequence_length, sequence_length))
        
        # Different bias patterns for different instruction types
        bias_patterns = {
            'BOUNDARY': self._boundary_bias_pattern,
            'SPEAKER_BOUNDARY': self._speaker_boundary_bias_pattern,
            'REFERENCE': self._reference_bias_pattern,
            'TOPIC_EVOLUTION': self._evolution_bias_pattern
        }
        
        # Apply appropriate bias pattern
        pattern_func = bias_patterns.get(instruction_type, self._default_bias_pattern)
        
        for pos in instruction_positions:
            bias = pattern_func(bias, pos, sequence_length)
        
        return bias
    
    def _boundary_bias_pattern(self, bias: 'torch.Tensor', pos: int, seq_len: int) -> 'torch.Tensor':
        """Create bias pattern for boundaries - reduce cross-boundary attention."""
        # Reduce attention across the boundary
        if pos > 0 and pos < seq_len:
            bias[pos-1:pos+1, pos+1:] -= 0.5
            bias[pos+1:, pos-1:pos+1] -= 0.5
        return bias
    
    def _speaker_boundary_bias_pattern(self, bias: 'torch.Tensor', pos: int, seq_len: int) -> 'torch.Tensor':
        """Create bias pattern for speaker boundaries - strong segmentation."""
        # Strong reduction across speaker boundaries
        if pos > 0 and pos < seq_len:
            bias[pos-1:pos+1, pos+1:] -= 1.0
            bias[pos+1:, pos-1:pos+1] -= 1.0
        return bias
    
    def _reference_bias_pattern(self, bias: 'torch.Tensor', pos: int, seq_len: int) -> 'torch.Tensor':
        """Create bias pattern for references - enhance long-range attention."""
        # Boost attention to earlier positions (references look backward)
        if pos > 0:
            bias[pos, :pos] += 0.3
        return bias
    
    def _evolution_bias_pattern(self, bias: 'torch.Tensor', pos: int, seq_len: int) -> 'torch.Tensor':
        """Create bias pattern for topic evolution - enhance local coherence."""
        # Boost attention in local neighborhood
        window = 3
        start = max(0, pos - window)
        end = min(seq_len, pos + window + 1)
        bias[start:end, start:end] += 0.2
        return bias
    
    def _default_bias_pattern(self, bias: 'torch.Tensor', pos: int, seq_len: int) -> 'torch.Tensor':
        """Default bias pattern - mild local enhancement."""
        # Slight boost to local attention
        if pos > 0 and pos < seq_len - 1:
            bias[pos-1:pos+2, pos-1:pos+2] += 0.1
        return bias
