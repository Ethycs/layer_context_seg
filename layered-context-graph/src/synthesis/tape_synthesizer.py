#!/usr/bin/env python3
"""
Tape Synthesizer - Actual Content Generation
============================================
This module performs the actual Graph → Tape₂ transformation,
generating NEW synthesized content (not just analysis reports).
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class TapeSynthesizer:
    """Synthesizes new documents from knowledge graphs"""
    
    def __init__(self):
        self.synthesis_strategies = {
            'executive_summary': self._synthesize_executive_summary,
            'tutorial': self._synthesize_tutorial,
            'reference': self._synthesize_reference,
            'readme': self._synthesize_readme
        }
    
    def synthesize(self, graph_data: Dict, strategy: str = 'executive_summary') -> Dict[str, Any]:
        """
        Generate actual new content from the knowledge graph
        
        This is the ACTUAL Graph → Tape₂ transformation that creates
        new synthesized documents, not analysis reports.
        """
        
        if strategy not in self.synthesis_strategies:
            available = list(self.synthesis_strategies.keys())
            raise ValueError(f"Unknown strategy: {strategy}. Available: {available}")
        
        # Extract components
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        original_text = graph_data.get('original_text', '')
        
        # Apply synthesis strategy
        synthesized_content = self.synthesis_strategies[strategy](nodes, edges, original_text)
        
        return {
            'tape2': synthesized_content,  # The actual synthesized content
            'strategy': strategy,
            'synthesis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'nodes_used': len(nodes),
                'edges_used': len(edges),
                'original_length': len(original_text),
                'synthesized_length': len(synthesized_content),
                'compression_ratio': len(original_text) / len(synthesized_content) if synthesized_content else 0
            }
        }
    
    def _synthesize_executive_summary(self, nodes: List[Dict], edges: List[Dict], original: str) -> str:
        """Generate concise executive summary"""
        
        # Filter for high-importance nodes
        important_nodes = sorted(
            nodes, 
            key=lambda n: n.get('importance', 0), 
            reverse=True
        )[:5]
        
        # Get foundational concepts
        foundational = [n for n in nodes if n.get('segment_type') == 'foundational'][:2]
        
        # Build summary
        summary = []
        summary.append("# Executive Summary\n")
        
        # Core concept
        if foundational:
            summary.append("## Core Concept\n")
            # Extract key insight from first foundational node
            key_concept = self._extract_key_sentence(foundational[0].get('content', ''))
            summary.append(f"{key_concept}\n\n")
        
        # Key Points
        summary.append("## Key Points\n")
        for i, node in enumerate(important_nodes[:3], 1):
            key_point = self._extract_key_sentence(node.get('content', ''))
            summary.append(f"{i}. {key_point}\n")
        
        # Technical highlights
        technical = [n for n in nodes if n.get('segment_type') == 'technical_core']
        if technical:
            summary.append("\n## Technical Approach\n")
            tech_summary = self._extract_key_sentence(technical[0].get('content', ''))
            summary.append(f"{tech_summary}\n")
        
        # Conclusion
        summary.append("\n## Summary\n")
        summary.append(f"This document analyzed {len(nodes)} interconnected concepts ")
        summary.append(f"forming a knowledge graph with {len(edges)} relationships.")
        
        return '\n'.join(summary)
    
    def _synthesize_tutorial(self, nodes: List[Dict], edges: List[Dict], original: str) -> str:
        """Generate step-by-step tutorial"""
        
        # Order nodes by dependencies
        ordered_nodes = self._topological_sort(nodes, edges)
        
        tutorial = []
        tutorial.append("# Step-by-Step Tutorial\n")
        
        # Introduction
        intro_node = next((n for n in nodes if 'introduction' in n.get('content', '').lower()), None)
        if intro_node:
            tutorial.append("## Introduction\n")
            tutorial.append(self._extract_key_sentence(intro_node['content']) + "\n\n")
        
        # Prerequisites
        foundational = [n for n in ordered_nodes if n.get('segment_type') == 'foundational']
        if foundational:
            tutorial.append("## Prerequisites\n")
            for node in foundational[:3]:
                tutorial.append(f"- {self._extract_key_sentence(node['content'])}\n")
            tutorial.append("\n")
        
        # Steps
        tutorial.append("## Steps\n")
        
        # Group nodes by type for logical flow
        technical_nodes = [n for n in ordered_nodes if n.get('segment_type') in ['technical_core', 'implementation']]
        
        for i, node in enumerate(technical_nodes[:10], 1):
            tutorial.append(f"### Step {i}: {self._generate_step_title(node)}\n")
            
            # Extract actionable content
            content = node.get('content', '')
            action = self._extract_action(content)
            tutorial.append(f"{action}\n\n")
            
            # Add example if available
            example_nodes = self._find_related_examples(node, nodes, edges)
            if example_nodes:
                tutorial.append("**Example:**\n")
                tutorial.append(f"```\n{self._extract_code_or_example(example_nodes[0]['content'])}\n```\n\n")
        
        # Summary
        tutorial.append("## Next Steps\n")
        application_nodes = [n for n in nodes if n.get('segment_type') == 'application']
        if application_nodes:
            tutorial.append(self._extract_key_sentence(application_nodes[0]['content']))
        
        return '\n'.join(tutorial)
    
    def _synthesize_reference(self, nodes: List[Dict], edges: List[Dict], original: str) -> str:
        """Generate reference documentation"""
        
        reference = []
        reference.append("# Reference Documentation\n")
        
        # Group by segment type
        grouped = {}
        for node in nodes:
            seg_type = node.get('segment_type', 'other')
            if seg_type not in grouped:
                grouped[seg_type] = []
            grouped[seg_type].append(node)
        
        # Table of Contents
        reference.append("## Table of Contents\n")
        for seg_type in ['foundational', 'technical_core', 'implementation', 'application']:
            if seg_type in grouped:
                reference.append(f"- [{self._humanize_segment_type(seg_type)}](#{seg_type})\n")
        reference.append("\n")
        
        # Sections
        for seg_type in ['foundational', 'technical_core', 'implementation', 'application']:
            if seg_type not in grouped:
                continue
                
            reference.append(f"## {self._humanize_segment_type(seg_type)}\n")
            
            for node in grouped[seg_type]:
                # Generate entry
                title = self._generate_reference_title(node)
                reference.append(f"### {title}\n")
                
                # Summary
                summary = self._extract_key_sentence(node['content'])
                reference.append(f"{summary}\n")
                
                # Details (if substantial content)
                if len(node['content']) > 200:
                    reference.append("\n**Details:**\n")
                    details = self._extract_details(node['content'])
                    reference.append(f"{details}\n")
                
                # Related concepts
                related = self._find_related_nodes(node['id'], nodes, edges)
                if related:
                    reference.append("\n**Related:** ")
                    reference.append(", ".join([f"`{r['id']}`" for r in related[:3]]))
                    reference.append("\n")
                
                reference.append("\n")
        
        return '\n'.join(reference)
    
    def _synthesize_readme(self, nodes: List[Dict], edges: List[Dict], original: str) -> str:
        """Generate README.md for the 'wiped' repository"""
        
        readme = []
        readme.append("# Project Documentation\n")
        readme.append("*Reconstructed from knowledge graph analysis*\n\n")
        
        # Project Overview
        foundational = [n for n in nodes if n.get('segment_type') == 'foundational']
        if foundational:
            readme.append("## Overview\n")
            overview = self._extract_key_sentence(foundational[0]['content'])
            readme.append(f"{overview}\n\n")
        
        # Key Features
        technical = [n for n in nodes if n.get('segment_type') == 'technical_core']
        if technical:
            readme.append("## Key Features\n")
            for node in technical[:5]:
                feature = self._extract_feature(node['content'])
                readme.append(f"- {feature}\n")
            readme.append("\n")
        
        # Architecture
        readme.append("## Architecture\n")
        arch_nodes = [n for n in nodes if 'architecture' in n.get('content', '').lower()]
        if arch_nodes:
            for node in arch_nodes[:3]:
                point = self._extract_key_sentence(node['content'])
                readme.append(f"- {point}\n")
        readme.append("\n")
        
        # Getting Started
        implementation = [n for n in nodes if n.get('segment_type') == 'implementation']
        if implementation:
            readme.append("## Getting Started\n")
            readme.append("```bash\n")
            # Extract any code/commands
            for node in implementation[:3]:
                if 'install' in node['content'].lower() or 'run' in node['content'].lower():
                    command = self._extract_command(node['content'])
                    if command:
                        readme.append(f"{command}\n")
            readme.append("```\n\n")
        
        # Documentation Structure
        readme.append("## Documentation\n")
        readme.append("This project contains the following key concepts:\n\n")
        
        # Group by type
        type_counts = {}
        for node in nodes:
            seg_type = node.get('segment_type', 'other')
            type_counts[seg_type] = type_counts.get(seg_type, 0) + 1
        
        for seg_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            readme.append(f"- **{self._humanize_segment_type(seg_type)}**: {count} components\n")
        
        return '\n'.join(readme)
    
    # Helper methods
    
    def _extract_key_sentence(self, content: str) -> str:
        """Extract the most important sentence from content"""
        if not content:
            return "No content available."
        
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if not sentences:
            return content[:100] + "..."
        
        # Prefer sentences with conclusion markers
        for marker in ['therefore', 'thus', 'in summary', 'the key', 'importantly']:
            for sent in sentences:
                if marker in sent.lower():
                    return sent + '.'
        
        # Otherwise return first substantial sentence
        for sent in sentences:
            if len(sent) > 20:
                return sent + '.'
        
        return sentences[0] + '.' if sentences else content[:100] + "..."
    
    def _extract_action(self, content: str) -> str:
        """Extract actionable instruction from content"""
        # Look for imperative sentences
        lines = content.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['create', 'add', 'implement', 'use', 'apply']):
                return line.strip()
        
        return self._extract_key_sentence(content)
    
    def _extract_feature(self, content: str) -> str:
        """Extract feature description"""
        # Look for feature-like descriptions
        if ':' in content:
            parts = content.split(':')
            if len(parts[0]) < 50:  # Likely a title
                return parts[0].strip()
        
        return self._extract_key_sentence(content).replace('.', '')
    
    def _extract_command(self, content: str) -> str:
        """Extract command from content"""
        lines = content.split('\n')
        for line in lines:
            if any(cmd in line for cmd in ['pip', 'npm', 'python', 'node', 'git']):
                return line.strip()
        return None
    
    def _generate_step_title(self, node: Dict) -> str:
        """Generate a title for a tutorial step"""
        content = node.get('content', '')
        
        # Try to extract a natural title
        if '\n' in content:
            first_line = content.split('\n')[0]
            if len(first_line) < 100:
                return first_line.strip('#').strip()
        
        # Generate from content type
        seg_type = node.get('segment_type', '')
        if seg_type == 'technical_core':
            return "Implement Core Functionality"
        elif seg_type == 'foundational':
            return "Understand the Concept"
        
        # Fallback
        words = content.split()[:5]
        return ' '.join(words)
    
    def _humanize_segment_type(self, seg_type: str) -> str:
        """Convert segment type to human-readable form"""
        mapping = {
            'foundational': 'Fundamental Concepts',
            'technical_core': 'Core Components',
            'implementation': 'Implementation Details',
            'application': 'Applications',
            'illustrative': 'Examples',
            'problem_solving': 'Solutions'
        }
        return mapping.get(seg_type, seg_type.replace('_', ' ').title())
    
    def _topological_sort(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """Sort nodes in dependency order"""
        # Simple implementation - in practice would use proper graph algorithms
        return sorted(nodes, key=lambda n: n.get('importance', 0), reverse=True)
    
    def _find_related_examples(self, node: Dict, all_nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """Find example nodes related to given node"""
        node_id = node['id']
        related = []
        
        for edge in edges:
            if edge['source'] == node_id or edge['target'] == node_id:
                other_id = edge['target'] if edge['source'] == node_id else edge['source']
                other_node = next((n for n in all_nodes if n['id'] == other_id), None)
                if other_node and other_node.get('segment_type') == 'illustrative':
                    related.append(other_node)
        
        return related
    
    def _find_related_nodes(self, node_id: str, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """Find nodes connected to given node"""
        related = []
        
        for edge in edges:
            if edge['source'] == node_id:
                related_node = next((n for n in nodes if n['id'] == edge['target']), None)
                if related_node:
                    related.append(related_node)
        
        return related
    
    def _extract_code_or_example(self, content: str) -> str:
        """Extract code snippet or example from content"""
        # Look for code blocks
        if '```' in content:
            parts = content.split('```')
            if len(parts) > 1:
                return parts[1].strip()
        
        # Look for indented content
        lines = content.split('\n')
        code_lines = [l for l in lines if l.startswith('    ') or l.startswith('\t')]
        if code_lines:
            return '\n'.join(code_lines)
        
        # Fallback
        return self._extract_key_sentence(content)
    
    def _extract_details(self, content: str) -> str:
        """Extract detailed explanation"""
        # Get middle sentences (not first or last)
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if len(sentences) > 2:
            return '. '.join(sentences[1:-1]) + '.'
        return content[:200] + "..."
    
    def _generate_reference_title(self, node: Dict) -> str:
        """Generate title for reference entry"""
        content = node.get('content', '')
        
        # Try to find a natural title
        if ':' in content:
            parts = content.split(':')
            if len(parts[0]) < 50:
                return parts[0].strip()
        
        # Use node ID as fallback
        return f"Component {node['id']}"