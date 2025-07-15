#!/usr/bin/env python3
"""
Enhanced Graph Reassembler - Rich Multi-Layered Reconstruction
==============================================================
This module performs sophisticated graph reassembly that preserves
technical content, code blocks, and creates rich documentation.
"""

from typing import Dict, List, Any, Optional, Tuple
import re
import json
from collections import defaultdict


class EnhancedGraphReassembler:
    """Enhanced reassembler for rich, multi-layered reconstruction"""
    
    def __init__(self):
        self.reassembly_rules = {
            'preserve_code_blocks': True,
            'maintain_technical_depth': True,
            'hierarchical_organization': True,
            'cross_reference_generation': True,
            'narrative_flow': True
        }
    
    def reassemble_graph(self, nodes: List[Dict], edges: List[Dict], 
                        original_document: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform rich reconstruction preserving technical content and structure
        """
        
        print(f"🔧 Enhanced reconstruction with {len(nodes)} nodes")
        
        # Analyze nodes for rich content
        analysis = self._analyze_rich_content(nodes)
        
        # Build hierarchical structure
        hierarchy = self._build_content_hierarchy(nodes, edges, analysis)
        
        # Generate rich reconstructed document
        reconstructed_text = self._generate_rich_reconstruction(hierarchy, nodes, edges)
        
        return {
            'graph': self._build_graph_structure(nodes, edges),
            'nodes': nodes,
            'edges': edges,
            'reassembled_text': reconstructed_text,
            'reconstruction_metadata': {
                'method': 'enhanced_multi_layered',
                'content_analysis': analysis,
                'hierarchy_depth': self._calculate_hierarchy_depth(hierarchy),
                'preserved_elements': {
                    'code_blocks': analysis['code_blocks'],
                    'technical_sections': analysis['technical_sections'],
                    'examples': analysis['examples']
                }
            }
        }
    
    def _analyze_rich_content(self, nodes: List[Dict]) -> Dict[str, Any]:
        """Analyze nodes for rich content types"""
        analysis = {
            'code_blocks': 0,
            'technical_sections': 0,
            'examples': 0,
            'theoretical_content': 0,
            'implementation_details': 0,
            'content_types': defaultdict(list)
        }
        
        for node in nodes:
            content = node.get('content', '')
            
            # Detect code blocks
            if '```' in content or 'def ' in content or 'class ' in content:
                analysis['code_blocks'] += 1
                analysis['content_types']['code'].append(node)
            
            # Detect technical content
            if any(term in content.lower() for term in ['implementation', 'algorithm', 'technical', 'architecture']):
                analysis['technical_sections'] += 1
                analysis['content_types']['technical'].append(node)
            
            # Detect examples
            if 'example' in content.lower() or '```python' in content:
                analysis['examples'] += 1
                analysis['content_types']['example'].append(node)
            
            # Detect theoretical content
            if any(term in content.lower() for term in ['theory', 'principle', 'concept', 'foundation']):
                analysis['theoretical_content'] += 1
                analysis['content_types']['theoretical'].append(node)
            
            # Implementation details
            if any(term in content.lower() for term in ['implementation', 'code', 'function', 'method']):
                analysis['implementation_details'] += 1
                analysis['content_types']['implementation'].append(node)
        
        return analysis
    
    def _build_content_hierarchy(self, nodes: List[Dict], edges: List[Dict], 
                                analysis: Dict) -> Dict[str, Any]:
        """Build hierarchical content structure"""
        
        hierarchy = {
            'technical_core': [],
            'foundations': [],
            'implementations': [],
            'examples': [],
            'discussions': [],
            'metadata': {}
        }
        
        # Categorize nodes into hierarchy levels
        for node in nodes:
            content = node.get('content', '')
            seg_type = node.get('segment_type', 'general')
            
            # Preserve full content with formatting
            enriched_node = {
                **node,
                'formatted_content': self._format_node_content(node),
                'has_code': self._has_code_block(content),
                'section_type': self._determine_section_type(content, seg_type)
            }
            
            # Assign to appropriate hierarchy level
            if seg_type == 'technical_core' or enriched_node['has_code']:
                hierarchy['technical_core'].append(enriched_node)
            elif seg_type == 'foundational':
                hierarchy['foundations'].append(enriched_node)
            elif seg_type == 'implementation':
                hierarchy['implementations'].append(enriched_node)
            elif seg_type == 'illustrative' or 'example' in content.lower():
                hierarchy['examples'].append(enriched_node)
            else:
                hierarchy['discussions'].append(enriched_node)
        
        # Sort each level by importance
        for level in hierarchy:
            if isinstance(hierarchy[level], list):
                hierarchy[level].sort(key=lambda x: x.get('importance', 0), reverse=True)
        
        return hierarchy
    
    def _generate_rich_reconstruction(self, hierarchy: Dict, nodes: List[Dict], 
                                     edges: List[Dict]) -> str:
        """Generate rich reconstructed document"""
        
        doc_parts = []
        
        # Header
        doc_parts.append("# Reconstructed Knowledge Structure")
        doc_parts.append("*Generated through iterative disassembly and layered reconstruction*\n")
        
        # Technical Core Section
        if hierarchy['technical_core']:
            doc_parts.append("## Technical Core\n")
            for i, node in enumerate(hierarchy['technical_core'], 1):
                doc_parts.append(self._format_technical_section(node, i))
        
        # Foundations Section
        if hierarchy['foundations']:
            doc_parts.append("\n## Foundational Concepts\n")
            for i, node in enumerate(hierarchy['foundations'], 1):
                doc_parts.append(self._format_foundation_section(node, i))
        
        # Implementation Details
        if hierarchy['implementations']:
            doc_parts.append("\n## Implementation Details\n")
            for i, node in enumerate(hierarchy['implementations'], 1):
                doc_parts.append(self._format_implementation_section(node, i))
        
        # Examples and Applications
        if hierarchy['examples']:
            doc_parts.append("\n## Examples and Applications\n")
            for i, node in enumerate(hierarchy['examples'], 1):
                doc_parts.append(self._format_example_section(node, i))
        
        # Cross-References and Relationships
        if edges:
            doc_parts.append("\n## Relationships and Cross-References\n")
            doc_parts.append(self._generate_relationship_map(nodes, edges))
        
        # Metadata and Analysis
        doc_parts.append("\n## Reconstruction Metadata\n")
        doc_parts.append(self._generate_metadata_section(hierarchy, nodes, edges))
        
        return '\n'.join(doc_parts)
    
    def _format_node_content(self, node: Dict) -> str:
        """Format node content preserving structure"""
        content = node.get('content', '')
        
        # Preserve code blocks
        if '```' in content:
            return content  # Already formatted
        
        # Check if it's code without markers
        if self._is_code_content(content):
            lines = content.split('\n')
            # Indent and wrap in code block
            return '```python\n' + '\n'.join(lines) + '\n```'
        
        return content
    
    def _has_code_block(self, content: str) -> bool:
        """Check if content contains code"""
        code_indicators = ['```', 'def ', 'class ', 'import ', 'from ', '{', '}', '()', '[]']
        return any(indicator in content for indicator in code_indicators)
    
    def _is_code_content(self, content: str) -> bool:
        """Determine if content is primarily code"""
        code_patterns = [
            r'def\s+\w+\s*\(',
            r'class\s+\w+',
            r'import\s+\w+',
            r'from\s+\w+\s+import',
            r'\w+\s*=\s*\w+',
            r'if\s+.*:',
            r'for\s+\w+\s+in\s+'
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, content):
                return True
        return False
    
    def _determine_section_type(self, content: str, seg_type: str) -> str:
        """Determine the section type for formatting"""
        content_lower = content.lower()
        
        if 'implementation' in content_lower or 'code' in content_lower:
            return 'implementation'
        elif 'example' in content_lower:
            return 'example'
        elif 'theory' in content_lower or 'concept' in content_lower:
            return 'theoretical'
        elif 'technical' in content_lower:
            return 'technical'
        else:
            return seg_type
    
    def _format_technical_section(self, node: Dict, index: int) -> str:
        """Format technical core section"""
        parts = []
        
        parts.append(f"### Technical Core - Section {index}\n")
        
        # Add metadata
        seg_type = node.get('segment_type', 'technical_core')
        importance = node.get('importance', 0)
        parts.append(f"*Type: {seg_type.title().replace('_', ' ')} | Importance: {importance:.1f}*\n")
        
        # Extract title if present
        content = node.get('content', '')
        title = self._extract_section_title(content)
        if title:
            parts.append(f"### {index}. **{title}**\n")
        
        # Add formatted content
        formatted_content = node.get('formatted_content', content)
        parts.append(formatted_content)
        
        # Add connections if available
        if 'connections' in node:
            parts.append("\n**Related Concepts:**")
            for conn in node['connections']:
                parts.append(f"- {conn}")
        
        parts.append("\n")
        return '\n'.join(parts)
    
    def _format_foundation_section(self, node: Dict, index: int) -> str:
        """Format foundational concept section"""
        parts = []
        
        parts.append(f"### Foundation {index}\n")
        
        content = node.get('content', '')
        title = self._extract_section_title(content)
        
        if title:
            parts.append(f"**{title}**\n")
        
        # Format content with emphasis on concepts
        parts.append(self._emphasize_key_concepts(content))
        
        parts.append("\n")
        return '\n'.join(parts)
    
    def _format_implementation_section(self, node: Dict, index: int) -> str:
        """Format implementation detail section"""
        parts = []
        
        parts.append(f"### Implementation {index}\n")
        
        content = node.get('formatted_content', node.get('content', ''))
        
        # Ensure code is properly formatted
        if not content.startswith('```') and self._is_code_content(content):
            content = f"```python\n{content}\n```"
        
        parts.append(content)
        parts.append("\n")
        
        return '\n'.join(parts)
    
    def _format_example_section(self, node: Dict, index: int) -> str:
        """Format example section"""
        parts = []
        
        parts.append(f"### Example {index}\n")
        
        content = node.get('content', '')
        
        # Extract description if present
        if '\n' in content:
            lines = content.split('\n')
            if not lines[0].startswith('```'):
                parts.append(f"*{lines[0]}*\n")
                content = '\n'.join(lines[1:])
        
        parts.append(content)
        parts.append("\n")
        
        return '\n'.join(parts)
    
    def _extract_section_title(self, content: str) -> Optional[str]:
        """Extract a title from content"""
        # Look for function/class definitions
        if match := re.search(r'def\s+(\w+)\s*\(', content):
            return f"`{match.group(1)}()` Function"
        elif match := re.search(r'class\s+(\w+)', content):
            return f"`{match.group(1)}` Class"
        
        # Look for markdown headers
        if match := re.search(r'^#+\s+(.+)$', content, re.MULTILINE):
            return match.group(1)
        
        # Look for emphasized text
        if match := re.search(r'\*\*(.+?)\*\*', content):
            return match.group(1)
        
        # Use first line if it's short
        first_line = content.split('\n')[0].strip()
        if len(first_line) < 80 and not first_line.endswith('.'):
            return first_line
        
        return None
    
    def _emphasize_key_concepts(self, content: str) -> str:
        """Emphasize key concepts in foundational content"""
        # Highlight technical terms
        technical_terms = [
            'percolation', 'graph', 'attention', 'context window',
            'partition', 'reconstruction', 'hierarchy', 'semantic'
        ]
        
        emphasized = content
        for term in technical_terms:
            emphasized = re.sub(
                rf'\b{term}\b', 
                f'**{term}**', 
                emphasized, 
                flags=re.IGNORECASE
            )
        
        return emphasized
    
    def _generate_relationship_map(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """Generate relationship visualization"""
        if not edges:
            return "*No explicit relationships found in the graph*\n"
        
        parts = []
        
        # Group edges by type
        edge_types = defaultdict(list)
        for edge in edges:
            edge_type = edge.get('type', 'connection')
            edge_types[edge_type].append(edge)
        
        # Format each type
        for edge_type, type_edges in edge_types.items():
            parts.append(f"\n**{edge_type.title()} Relationships:**")
            
            for edge in type_edges[:5]:  # Show top 5
                source_node = next((n for n in nodes if n['id'] == edge['source']), None)
                target_node = next((n for n in nodes if n['id'] == edge['target']), None)
                
                if source_node and target_node:
                    source_desc = self._get_node_description(source_node)
                    target_desc = self._get_node_description(target_node)
                    weight = edge.get('weight', 1.0)
                    
                    parts.append(f"- {source_desc} → {target_desc} (strength: {weight:.2f})")
        
        return '\n'.join(parts)
    
    def _get_node_description(self, node: Dict) -> str:
        """Get concise node description"""
        content = node.get('content', '')
        
        # Try to get title
        if title := self._extract_section_title(content):
            return title[:50]
        
        # Use first few words
        words = content.split()[:5]
        return ' '.join(words) + '...'
    
    def _generate_metadata_section(self, hierarchy: Dict, nodes: List[Dict], 
                                  edges: List[Dict]) -> str:
        """Generate metadata section"""
        parts = []
        
        # Content statistics
        parts.append("### Content Statistics\n")
        parts.append(f"- Total Nodes: {len(nodes)}")
        parts.append(f"- Total Relationships: {len(edges)}")
        parts.append(f"- Technical Core Sections: {len(hierarchy['technical_core'])}")
        parts.append(f"- Foundational Concepts: {len(hierarchy['foundations'])}")
        parts.append(f"- Implementation Details: {len(hierarchy['implementations'])}")
        parts.append(f"- Examples: {len(hierarchy['examples'])}")
        
        # Node type distribution
        node_types = defaultdict(int)
        for node in nodes:
            seg_type = node.get('segment_type', 'general')
            node_types[seg_type] += 1
        
        parts.append("\n### Node Type Distribution\n")
        for seg_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(nodes)) * 100
            parts.append(f"- {seg_type}: {count} ({percentage:.1f}%)")
        
        # Code content analysis
        code_nodes = sum(1 for n in nodes if self._has_code_block(n.get('content', '')))
        parts.append(f"\n### Code Content\n")
        parts.append(f"- Nodes with code: {code_nodes}")
        parts.append(f"- Code percentage: {(code_nodes/len(nodes)*100):.1f}%")
        
        return '\n'.join(parts)
    
    def _build_graph_structure(self, nodes: List[Dict], edges: List[Dict]) -> Dict:
        """Build basic graph structure"""
        graph = {}
        
        for node in nodes:
            graph[node['id']] = {
                'content': node['content'],
                'type': node.get('segment_type', 'general'),
                'importance': node.get('importance', 0),
                'connections': []
            }
        
        for edge in edges:
            if edge['source'] in graph:
                graph[edge['source']]['connections'].append({
                    'target': edge['target'],
                    'type': edge.get('type', 'connection'),
                    'weight': edge.get('weight', 1.0)
                })
        
        return graph
    
    def _calculate_hierarchy_depth(self, hierarchy: Dict) -> int:
        """Calculate depth of hierarchy"""
        non_empty_levels = sum(
            1 for level in ['technical_core', 'foundations', 'implementations', 'examples', 'discussions']
            if hierarchy.get(level)
        )
        return non_empty_levels