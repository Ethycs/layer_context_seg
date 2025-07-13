"""
Seed-Based Reconstructor: Uses original document summary as scaffold for condensation
This approach prevents duplication and preserves technical detail by using a detailed 
summary of the original document as a guide for the condensed version.
"""

import json
import os
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict, Counter
import re
import hashlib


class SeedBasedReconstructor:
    """
    Uses a detailed summary of the original document as a scaffold/seed for condensation.
    This prevents duplication and ensures the condensed version stays grounded in the original.
    """
    
    def __init__(self, 
                 similarity_threshold=0.4, 
                 dedup_threshold=0.8,
                 max_section_length=1500,
                 preserve_code_blocks=True):
        self.similarity_threshold = similarity_threshold
        self.dedup_threshold = dedup_threshold  # Higher threshold for deduplication
        self.max_section_length = max_section_length
        self.preserve_code_blocks = preserve_code_blocks
        
        # Track content to prevent duplication
        self.seen_content_hashes = set()
        self.preserved_code_blocks = []
        self.technical_concepts = []
        self.key_insights = []
        
    def reconstruct_from_graph(self, graph_file_path: str, original_text: str, original_filename: str) -> str:
        """
        Main method: reconstruct condensed document using original text as seed
        
        Args:
            graph_file_path: Path to the graph JSON file
            original_text: Original document text to use as seed/scaffold
            original_filename: Original filename for output naming
            
        Returns:
            Condensed markdown content grounded in original document
        """
        
        # Load graph data
        with open(graph_file_path, 'r') as f:
            graph_data = json.load(f)
        
        print(f"🌱 Starting seed-based reconstruction with original text ({len(original_text)} chars)")
        print(f"   📊 Processing {len(graph_data.get('nodes', []))} nodes from graph")
        
        # Step 1: Create detailed summary scaffold from original text
        summary_scaffold = self._create_summary_scaffold(original_text)
        print(f"   📝 Created summary scaffold with {len(summary_scaffold)} sections")
        
        # Step 2: Extract and deduplicate key insights from graph nodes
        unique_insights = self._extract_unique_insights(graph_data['nodes'])
        print(f"   💡 Extracted {len(unique_insights)} unique insights")
        
        # Step 3: Extract and preserve all code blocks and technical content
        preserved_technical = self._extract_technical_content(graph_data['nodes'])
        print(f"   🔧 Preserved {len(preserved_technical['code_blocks'])} code blocks and {len(preserved_technical['technical_concepts'])} concepts")
        
        # Step 4: Map insights to scaffold sections
        enriched_scaffold = self._enrich_scaffold_with_insights(summary_scaffold, unique_insights, preserved_technical)
        print(f"   🔗 Enriched scaffold with graph insights")
        
        # Step 5: Generate final condensed document
        condensed_content = self._generate_final_document(enriched_scaffold, original_filename)
        print(f"   ✅ Generated seed-based condensed document ({len(condensed_content)} chars)")
        
        return condensed_content
    
    def _create_summary_scaffold(self, original_text: str) -> List[Dict]:
        """
        Create a detailed summary scaffold from the original document
        This serves as the backbone/seed for the condensed version
        """
        # Split into logical sections
        sections = self._split_into_logical_sections(original_text)
        
        scaffold = []
        for i, section in enumerate(sections):
            # Create summary for each section
            section_summary = self._summarize_section(section)
            
            scaffold.append({
                'section_id': f"scaffold_{i}",
                'title': section_summary['title'],
                'summary': section_summary['summary'],
                'key_points': section_summary['key_points'],
                'original_length': len(section),
                'contains_code': self._contains_code(section),
                'technical_level': self._assess_technical_level(section),
                'original_section': section  # Keep for reference
            })
        
        return scaffold
    
    def _split_into_logical_sections(self, text: str) -> List[str]:
        """
        Split text into logical sections based on structure markers
        """
        # Look for clear section breaks
        section_patterns = [
            r'\n#{1,3}\s+[^\n]+\n',  # Markdown headers
            r'\n\d+\.\s+[^\n]+\n',   # Numbered sections
            r'\n[A-Z][A-Z\s]+:\n',   # ALL CAPS headers
            r'\n\*\*[^*]+\*\*\n',    # Bold headers
            r'\n---+\n',             # Horizontal rules
        ]
        
        # Find all potential split points
        split_points = [0]
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text))
            split_points.extend([m.start() for m in matches])
        
        # Sort and deduplicate
        split_points = sorted(set(split_points))
        split_points.append(len(text))
        
        # Create sections
        sections = []
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            section = text[start:end].strip()
            
            if len(section) > 100:  # Only keep substantial sections
                sections.append(section)
        
        # If no clear sections found, split by length
        if len(sections) < 2:
            sections = self._split_by_length(text, target_length=2000)
        
        return sections
    
    def _split_by_length(self, text: str, target_length: int = 2000) -> List[str]:
        """Split text into chunks of approximately target_length"""
        sections = []
        current_pos = 0
        
        while current_pos < len(text):
            end_pos = min(current_pos + target_length, len(text))
            
            # Try to break at sentence boundary
            if end_pos < len(text):
                # Look for sentence end within last 200 chars
                search_start = max(end_pos - 200, current_pos)
                sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+', text[search_start:end_pos])]
                
                if sentence_ends:
                    end_pos = search_start + sentence_ends[-1]
            
            section = text[current_pos:end_pos].strip()
            if section:
                sections.append(section)
            
            current_pos = end_pos
        
        return sections
    
    def _summarize_section(self, section: str) -> Dict:
        """
        Create a detailed summary of a section
        """
        # Extract title from first line or header
        lines = section.split('\n')
        title = "Section"
        
        for line in lines[:3]:  # Check first few lines
            line = line.strip()
            if line:
                # Check if it looks like a header
                if (line.startswith('#') or 
                    line.isupper() or 
                    re.match(r'^\d+\.', line) or
                    re.match(r'^\*\*.*\*\*$', line)):
                    title = re.sub(r'[#*\d\.\s]+', '', line).strip()
                    break
                elif len(line) < 100:  # Short lines might be titles
                    title = line
                    break
        
        # Extract key sentences for summary
        sentences = re.split(r'[.!?]+', section)
        key_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Substantial sentences
                # Prioritize sentences with key terms
                if any(term in sentence.lower() for term in 
                      ['algorithm', 'method', 'approach', 'process', 'system', 
                       'implementation', 'result', 'conclusion', 'important']):
                    key_sentences.append(sentence)
                elif len(key_sentences) < 3:  # Keep first few sentences
                    key_sentences.append(sentence)
        
        # Limit to 3 key sentences for conciseness
        key_sentences = key_sentences[:3]
        
        return {
            'title': title,
            'summary': '. '.join(key_sentences) + '.' if key_sentences else section[:200] + '...',
            'key_points': self._extract_key_points(section),
        }
    
    def _extract_key_points(self, section: str) -> List[str]:
        """Extract key bullet points or numbered items from section"""
        key_points = []
        
        # Look for bullet points
        bullet_matches = re.findall(r'^\s*[-*•]\s+(.+)$', section, re.MULTILINE)
        key_points.extend([point.strip() for point in bullet_matches])
        
        # Look for numbered items
        numbered_matches = re.findall(r'^\s*\d+\.\s+(.+)$', section, re.MULTILINE)
        key_points.extend([point.strip() for point in numbered_matches])
        
        # Limit to most important points
        return key_points[:5]
    
    def _contains_code(self, text: str) -> bool:
        """Check if text contains code blocks or inline code"""
        return bool(re.search(r'```|`[^`]+`|def |class |import |function\s*\(', text))
    
    def _assess_technical_level(self, text: str) -> str:
        """Assess the technical complexity level of text"""
        technical_terms = ['algorithm', 'implementation', 'class', 'function', 'method', 
                          'parameter', 'variable', 'object', 'module', 'library']
        
        term_count = sum(1 for term in technical_terms if term in text.lower())
        
        if term_count > 5:
            return 'high'
        elif term_count > 2:
            return 'medium'
        else:
            return 'low'
    
    def _extract_unique_insights(self, nodes: List[Dict]) -> List[Dict]:
        """
        Extract unique insights from graph nodes, removing duplicates
        """
        insights = []
        content_hashes = set()
        
        for node in nodes:
            content = self._extract_node_content(node)
            if not content or len(content) < 50:  # Skip very short content
                continue
            
            # Create content hash for deduplication
            content_hash = hashlib.md5(self._normalize_content(content).encode()).hexdigest()
            
            if content_hash not in content_hashes:
                content_hashes.add(content_hash)
                
                insights.append({
                    'content': content,
                    'node_id': node.get('id', 'unknown'),
                    'type': node.get('type', 'general'),
                    'weight': node.get('weight', 1.0),
                    'contains_code': self._contains_code(content),
                    'technical_level': self._assess_technical_level(content),
                    'hash': content_hash
                })
        
        # Sort by weight and technical importance
        insights.sort(key=lambda x: (x['weight'], 
                                   1 if x['contains_code'] else 0,
                                   1 if x['technical_level'] == 'high' else 0), 
                     reverse=True)
        
        return insights
    
    def _extract_node_content(self, node: Dict) -> str:
        """Extract content from a graph node"""
        for field in ['content', 'text', 'original_text', 'data']:
            if field in node:
                content = node[field]
                if isinstance(content, str):
                    return content.strip()
                elif isinstance(content, dict):
                    # Try to extract text from nested structure
                    for subfield in ['text', 'content', 'value']:
                        if subfield in content:
                            return str(content[subfield]).strip()
        return str(node).strip()
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for deduplication comparison"""
        # Remove extra whitespace and formatting
        normalized = re.sub(r'\s+', ' ', content)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return normalized.lower().strip()
    
    def _extract_technical_content(self, nodes: List[Dict]) -> Dict:
        """
        Extract and preserve all technical content (code blocks, concepts, etc.)
        """
        technical_content = {
            'code_blocks': [],
            'technical_concepts': [],
            'algorithms': [],
            'data_structures': []
        }
        
        seen_code_hashes = set()
        
        for node in nodes:
            content = self._extract_node_content(node)
            
            # Extract code blocks
            code_blocks = re.findall(r'```[\s\S]*?```', content)
            for code in code_blocks:
                code_hash = hashlib.md5(code.encode()).hexdigest()
                if code_hash not in seen_code_hashes:
                    seen_code_hashes.add(code_hash)
                    technical_content['code_blocks'].append({
                        'code': code,
                        'node_id': node.get('id', 'unknown'),
                        'hash': code_hash
                    })
            
            # Extract inline code
            inline_code = re.findall(r'`([^`]+)`', content)
            for code in inline_code:
                if len(code) > 5:  # Only substantial inline code
                    technical_content['technical_concepts'].append(code)
            
            # Extract algorithm mentions
            if 'algorithm' in content.lower():
                alg_sentences = [s for s in re.split(r'[.!?]', content) 
                               if 'algorithm' in s.lower()]
                technical_content['algorithms'].extend(alg_sentences)
        
        # Deduplicate technical concepts
        technical_content['technical_concepts'] = list(set(technical_content['technical_concepts']))
        technical_content['algorithms'] = list(set(technical_content['algorithms']))
        
        return technical_content
    
    def _enrich_scaffold_with_insights(self, scaffold: List[Dict], insights: List[Dict], 
                                     technical_content: Dict) -> List[Dict]:
        """
        Enrich the summary scaffold with unique insights from the graph
        """
        enriched_scaffold = []
        
        for section in scaffold:
            enriched_section = section.copy()
            
            # Find relevant insights for this section
            relevant_insights = self._find_relevant_insights(section, insights)
            
            # Add relevant technical content
            relevant_technical = self._find_relevant_technical(section, technical_content)
            
            # Enrich the section
            enriched_section['insights'] = relevant_insights
            enriched_section['technical_content'] = relevant_technical
            enriched_section['enhanced_summary'] = self._create_enhanced_summary(
                section, relevant_insights, relevant_technical)
            
            enriched_scaffold.append(enriched_section)
        
        return enriched_scaffold
    
    def _find_relevant_insights(self, section: Dict, insights: List[Dict]) -> List[Dict]:
        """Find insights that are relevant to a scaffold section"""
        section_text = section['summary'] + ' ' + ' '.join(section['key_points'])
        section_words = set(self._normalize_content(section_text).split())
        
        relevant = []
        for insight in insights:
            insight_words = set(self._normalize_content(insight['content']).split())
            
            # Calculate overlap
            overlap = len(section_words.intersection(insight_words))
            total_words = len(section_words.union(insight_words))
            similarity = overlap / total_words if total_words > 0 else 0
            
            if similarity > self.similarity_threshold:
                insight_copy = insight.copy()
                insight_copy['relevance_score'] = similarity
                relevant.append(insight_copy)
        
        # Sort by relevance and return top matches
        relevant.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant[:3]  # Limit to top 3 to avoid overwhelming
    
    def _find_relevant_technical(self, section: Dict, technical_content: Dict) -> Dict:
        """Find technical content relevant to a scaffold section"""
        section_text = section.get('original_section', section['summary'])
        
        relevant_technical = {
            'code_blocks': [],
            'concepts': [],
            'algorithms': []
        }
        
        # Find relevant code blocks
        for code_item in technical_content['code_blocks']:
            if any(word in section_text.lower() for word in 
                  ['function', 'class', 'def', 'algorithm', 'implementation']):
                relevant_technical['code_blocks'].append(code_item)
        
        # Find relevant concepts
        section_lower = section_text.lower()
        for concept in technical_content['technical_concepts']:
            if any(word in concept.lower() for word in section_lower.split()):
                relevant_technical['concepts'].append(concept)
        
        # Find relevant algorithms
        for alg in technical_content['algorithms']:
            if any(word in section_text.lower() for word in alg.lower().split()):
                relevant_technical['algorithms'].append(alg)
        
        return relevant_technical
    
    def _create_enhanced_summary(self, section: Dict, insights: List[Dict], 
                               technical_content: Dict) -> str:
        """
        Create an enhanced summary by combining scaffold with insights
        """
        enhanced = section['summary']
        
        # Add key insights
        if insights:
            enhanced += "\n\nKey insights from analysis:"
            for insight in insights[:2]:  # Limit to avoid bloat
                # Extract the most important sentence from insight
                sentences = re.split(r'[.!?]', insight['content'])
                best_sentence = max(sentences, key=len) if sentences else insight['content']
                enhanced += f"\n- {best_sentence.strip()}"
        
        # Add technical content if relevant
        if technical_content['concepts']:
            enhanced += f"\n\nTechnical concepts: {', '.join(technical_content['concepts'][:5])}"
        
        return enhanced
    
    def _generate_final_document(self, enriched_scaffold: List[Dict], original_filename: str) -> str:
        """
        Generate the final condensed document from enriched scaffold
        """
        doc_parts = []
        
        # Header
        doc_parts.append(f"# Condensed Analysis: {original_filename}")
        doc_parts.append(f"*Generated using seed-based reconstruction to preserve original structure and prevent duplication*\n")
        
        # Executive Summary
        doc_parts.append("## Executive Summary")
        summary_points = []
        for section in enriched_scaffold[:3]:  # Use first 3 sections for summary
            summary_points.append(f"- {section['title']}: {section['summary'][:150]}...")
        doc_parts.append('\n'.join(summary_points))
        doc_parts.append("")
        
        # Main content sections
        for i, section in enumerate(enriched_scaffold, 1):
            doc_parts.append(f"## {i}. {section['title']}")
            doc_parts.append(section['enhanced_summary'])
            
            # Add key points if available
            if section.get('key_points'):
                doc_parts.append("\n**Key Points:**")
                for point in section['key_points'][:3]:
                    doc_parts.append(f"- {point}")
            
            # Add relevant code blocks
            if section.get('technical_content', {}).get('code_blocks'):
                doc_parts.append("\n**Technical Implementation:**")
                for code_item in section['technical_content']['code_blocks'][:1]:  # Limit to 1 per section
                    doc_parts.append(code_item['code'])
            
            doc_parts.append("")  # Section separator
        
        # Technical appendix with preserved code
        all_code_blocks = []
        for section in enriched_scaffold:
            all_code_blocks.extend(section.get('technical_content', {}).get('code_blocks', []))
        
        if all_code_blocks and len(all_code_blocks) > 3:
            doc_parts.append("## Technical Appendix")
            doc_parts.append("### Preserved Code Blocks")
            
            seen_hashes = set()
            for code_item in all_code_blocks:
                if code_item['hash'] not in seen_hashes:
                    seen_hashes.add(code_item['hash'])
                    doc_parts.append(code_item['code'])
                    doc_parts.append("")
        
        return '\n'.join(doc_parts)


def main():
    """Test the seed-based reconstructor"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python seed_based_reconstructor.py <graph_file> <original_text_file>")
        return
    
    graph_file = sys.argv[1]
    original_file = sys.argv[2]
    
    # Read original text
    with open(original_file, 'r') as f:
        original_text = f.read()
    
    # Create reconstructor
    reconstructor = SeedBasedReconstructor()
    
    # Generate condensed document
    condensed = reconstructor.reconstruct_from_graph(
        graph_file, original_text, os.path.basename(original_file))
    
    # Save result
    output_file = original_file.replace('.txt', '_seed_condensed.md')
    with open(output_file, 'w') as f:
        f.write(condensed)
    
    print(f"✅ Seed-based condensed document saved to: {output_file}")


if __name__ == "__main__":
    main()
