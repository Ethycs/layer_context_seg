"""
Content Deduplicator: Removes duplicate content from reorganized outputs
Fixes the massive duplication issues in the current reorganized documents
"""

import re
import hashlib
from typing import List, Dict, Set, Tuple
from collections import defaultdict


class ContentDeduplicator:
    """
    Removes duplicate content from markdown documents while preserving structure
    """
    
    def __init__(self, similarity_threshold=0.85, min_content_length=100):
        self.similarity_threshold = similarity_threshold
        self.min_content_length = min_content_length
        self.seen_content_hashes = set()
        self.duplicate_count = 0
        
    def deduplicate_markdown(self, content: str) -> str:
        """
        Remove duplicate content from a markdown document
        
        Args:
            content: The markdown content to deduplicate
            
        Returns:
            Deduplicated markdown content
        """
        print(f"🔄 Starting deduplication process...")
        
        # Split into sections (by headers)
        sections = self._split_into_sections(content)
        print(f"   📊 Found {len(sections)} sections")
        
        # Deduplicate sections
        unique_sections = self._deduplicate_sections(sections)
        print(f"   ✂️  Removed {self.duplicate_count} duplicate sections")
        
        # Deduplicate code blocks within sections
        deduplicated_sections = self._deduplicate_code_blocks(unique_sections)
        print(f"   🔧 Deduplicated code blocks")
        
        # Reassemble document
        result = self._reassemble_document(deduplicated_sections)
        
        compression_ratio = len(result) / len(content) if content else 1.0
        print(f"   ✅ Deduplication complete. Compression ratio: {compression_ratio:.2f}")
        
        return result
    
    def _split_into_sections(self, content: str) -> List[Dict]:
        """
        Split markdown content into sections based on headers
        """
        sections = []
        
        # Find all headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')
        
        current_section = {
            'level': 0,
            'title': 'Preamble',
            'content': '',
            'start_line': 0
        }
        
        for i, line in enumerate(lines):
            header_match = re.match(header_pattern, line)
            
            if header_match:
                # Save previous section
                if current_section['content'].strip():
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                current_section = {
                    'level': level,
                    'title': title,
                    'content': line + '\n',
                    'start_line': i
                }
            else:
                current_section['content'] += line + '\n'
        
        # Add final section
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    def _deduplicate_sections(self, sections: List[Dict]) -> List[Dict]:
        """
        Remove duplicate sections based on content similarity
        """
        unique_sections = []
        content_signatures = set()
        
        for section in sections:
            # Create content signature for comparison
            signature = self._create_content_signature(section['content'])
            
            # Check for duplicates
            is_duplicate = False
            for existing_sig in content_signatures:
                if self._calculate_signature_similarity(signature, existing_sig) > self.similarity_threshold:
                    is_duplicate = True
                    self.duplicate_count += 1
                    print(f"     🗑️  Removing duplicate section: '{section['title'][:50]}...'")
                    break
            
            if not is_duplicate:
                content_signatures.add(signature)
                unique_sections.append(section)
            
        return unique_sections
    
    def _create_content_signature(self, content: str) -> str:
        """
        Create a signature for content comparison (normalized hash)
        """
        # Normalize content for comparison
        normalized = self._normalize_content_for_comparison(content)
        
        # Create hash
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _normalize_content_for_comparison(self, content: str) -> str:
        """
        Normalize content for duplicate detection
        """
        # Remove headers
        content = re.sub(r'^#{1,6}\s+.+$', '', content, flags=re.MULTILINE)
        
        # Remove code blocks temporarily (we'll handle them separately)
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        content = re.sub(r'```[\s\S]*?```', '<CODE_BLOCK>', content)
        
        # Remove markdown formatting
        content = re.sub(r'[*_`#\[\]()]', '', content)
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common filler words
        filler_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        words = content.lower().split()
        words = [w for w in words if w not in filler_words and len(w) > 2]
        
        normalized = ' '.join(words)
        
        # Add back code block indicators
        for i, _ in enumerate(code_blocks):
            normalized += f' CODE_BLOCK_{i}'
        
        return normalized.strip()
    
    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """
        Calculate similarity between two content signatures
        """
        # For now, just check if they're identical (could be enhanced with fuzzy matching)
        return 1.0 if sig1 == sig2 else 0.0
    
    def _deduplicate_code_blocks(self, sections: List[Dict]) -> List[Dict]:
        """
        Remove duplicate code blocks within sections
        """
        deduplicated_sections = []
        global_code_signatures = set()
        
        for section in sections:
            content = section['content']
            
            # Find all code blocks
            code_blocks = re.findall(r'```[\s\S]*?```', content)
            
            # Track which code blocks to keep
            unique_code_blocks = []
            
            for code_block in code_blocks:
                code_signature = self._create_code_signature(code_block)
                
                if code_signature not in global_code_signatures:
                    global_code_signatures.add(code_signature)
                    unique_code_blocks.append(code_block)
                else:
                    print(f"     🗑️  Removing duplicate code block ({len(code_block)} chars)")
            
            # Replace code blocks in content
            new_content = content
            for original_code in code_blocks:
                if original_code in unique_code_blocks:
                    continue  # Keep this one
                else:
                    # Remove this duplicate code block
                    new_content = new_content.replace(original_code, '', 1)
            
            # Clean up extra whitespace
            new_content = re.sub(r'\n{3,}', '\n\n', new_content)
            
            section_copy = section.copy()
            section_copy['content'] = new_content
            deduplicated_sections.append(section_copy)
        
        return deduplicated_sections
    
    def _create_code_signature(self, code_block: str) -> str:
        """
        Create a signature for code block comparison
        """
        # Extract just the code content (remove ``` markers)
        code_content = re.sub(r'^```.*$', '', code_block, flags=re.MULTILINE)
        code_content = re.sub(r'^```$', '', code_content, flags=re.MULTILINE)
        
        # Normalize whitespace and comments
        code_content = re.sub(r'#.*$', '', code_content, flags=re.MULTILINE)  # Remove comments
        code_content = re.sub(r'\s+', ' ', code_content)  # Normalize whitespace
        code_content = code_content.strip()
        
        return hashlib.md5(code_content.encode()).hexdigest()
    
    def _reassemble_document(self, sections: List[Dict]) -> str:
        """
        Reassemble the deduplicated sections into a coherent document
        """
        result_parts = []
        
        for section in sections:
            # Clean up the section content
            content = section['content'].strip()
            
            # Ensure proper spacing between sections
            if result_parts and not content.startswith('#'):
                result_parts.append('')  # Add blank line before non-header sections
            
            result_parts.append(content)
        
        # Join and clean up final formatting
        result = '\n'.join(result_parts)
        
        # Remove excessive blank lines
        result = re.sub(r'\n{4,}', '\n\n\n', result)
        
        return result.strip()


def deduplicate_file(input_file: str, output_file: str = None) -> str:
    """
    Deduplicate a markdown file
    
    Args:
        input_file: Path to input markdown file
        output_file: Path for deduplicated output (optional)
        
    Returns:
        Path to deduplicated file
    """
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"📖 Reading {input_file} ({len(content)} characters)")
    
    # Deduplicate
    deduplicator = ContentDeduplicator()
    deduplicated_content = deduplicator.deduplicate_markdown(content)
    
    # Determine output file
    if output_file is None:
        base_name = input_file.rsplit('.', 1)[0]
        output_file = f"{base_name}_deduplicated.md"
    
    # Save deduplicated content
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(deduplicated_content)
    
    print(f"💾 Saved deduplicated content to {output_file}")
    print(f"📊 Size reduction: {len(content)} → {len(deduplicated_content)} characters")
    
    return output_file


def main():
    """Test the deduplicator"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python content_deduplicator.py <input_file> [output_file]")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    result_file = deduplicate_file(input_file, output_file)
    print(f"✅ Deduplication complete: {result_file}")


if __name__ == "__main__":
    main()
