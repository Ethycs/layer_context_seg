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
            "ANALYZE": "Look for causal relationships",
            "CONNECT": "Link related concepts across time",
            "ABSTRACT": "Extract high-level themes",
            "BOUNDARY": "Detect topic or section changes",
            "CLASSIFY": "Group similar content together",
            "PRIORITIZE": "Focus on important information",
            "CODE": "Preserve code structure and semantics",
            "MATH": "Maintain mathematical relationships",
            "NARRATIVE": "Follow story or argument flow",
            "TECHNICAL": "Preserve technical details"
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
