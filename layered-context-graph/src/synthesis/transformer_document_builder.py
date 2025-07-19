#!/usr/bin/env python3
"""
Transformer Document Builder - Progressive Document Generation from Graphs
========================================================================
Uses transformer's natural autoregressive architecture to progressively
build coherent documents from knowledge graphs, rather than tree-based assembly.
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
import logging
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class GenerationContext:
    """Tracks the current generation state"""
    generated_text: str
    visited_nodes: set
    current_section: str
    section_depth: int
    remaining_nodes: deque
    

class TransformerDocumentBuilder:
    """
    Builds documents progressively using transformer generation
    rather than hierarchical tree assembly
    """
    
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generation strategies
        self.generation_strategies = {
            'linear': self._generate_linear,
            'guided': self._generate_guided,
            'interactive': self._generate_interactive,
            'narrative': self._generate_narrative
        }
    
    def build_document(self, graph_data: Dict, strategy: str = 'guided') -> str:
        """
        Progressively build a document from graph data
        
        Args:
            graph_data: Contains nodes, edges, and metadata
            strategy: Generation strategy to use
            
        Returns:
            Generated document text
        """
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        # Initialize generation context
        context = self._initialize_context(nodes, edges)
        
        # Select generation strategy
        generate_fn = self.generation_strategies.get(strategy, self._generate_guided)
        
        # Generate document progressively
        document = generate_fn(context, nodes, edges)
        
        return document
    
    def _initialize_context(self, nodes: List[Dict], edges: List[Dict]) -> GenerationContext:
        """Initialize the generation context"""
        # Sort nodes by importance for initial ordering
        sorted_nodes = sorted(nodes, key=lambda n: n.get('importance', 0), reverse=True)
        
        return GenerationContext(
            generated_text="",
            visited_nodes=set(),
            current_section="introduction",
            section_depth=0,
            remaining_nodes=deque(sorted_nodes)
        )
    
    def _generate_guided(self, context: GenerationContext, 
                        nodes: List[Dict], edges: List[Dict]) -> str:
        """
        Generate document with guided progression through the graph
        """
        document_parts = []
        
        # Start with document seed
        seed_text = self._create_document_seed(nodes, edges)
        document_parts.append(seed_text)
        
        # Build node index for quick lookup
        node_map = {node['id']: node for node in nodes}
        
        # Create adjacency list from edges
        graph = self._build_adjacency_list(nodes, edges)
        
        # Progressive generation
        while context.remaining_nodes:
            # Select next node based on context
            next_node = self._select_next_node(
                context, 
                node_map, 
                graph
            )
            
            if not next_node:
                break
                
            # Generate content for this node
            node_content = self._generate_node_content(
                next_node,
                context,
                node_map,
                graph
            )
            
            # Add to document with appropriate transition
            transition = self._generate_transition(
                context.generated_text,
                node_content,
                next_node
            )
            
            document_parts.append(transition)
            document_parts.append(node_content)
            
            # Update context
            context.generated_text = '\n'.join(document_parts)
            context.visited_nodes.add(next_node['id'])
            context.remaining_nodes.remove(next_node)
            
            # Check for section transitions
            if self._should_start_new_section(context, next_node):
                section_header = self._generate_section_header(context, next_node)
                document_parts.append(f"\n{section_header}\n")
                context.current_section = next_node.get('segment_type', 'content')
        
        # Generate conclusion
        conclusion = self._generate_conclusion(context, nodes)
        document_parts.append(conclusion)
        
        return '\n'.join(document_parts)
    
    def _generate_linear(self, context: GenerationContext,
                        nodes: List[Dict], edges: List[Dict]) -> str:
        """
        Simple linear generation following importance order
        """
        document_parts = []
        
        # Introduction
        intro = "# Document Overview\n\n"
        intro += self._summarize_graph_content(nodes, edges)
        document_parts.append(intro)
        
        # Process nodes in importance order
        for node in context.remaining_nodes:
            content = self._format_node_for_linear(node)
            document_parts.append(content)
            context.visited_nodes.add(node['id'])
        
        return '\n\n'.join(document_parts)
    
    def _generate_narrative(self, context: GenerationContext,
                           nodes: List[Dict], edges: List[Dict]) -> str:
        """
        Generate a narrative flow through the graph
        """
        document_parts = []
        
        # Find narrative threads
        threads = self._identify_narrative_threads(nodes, edges)
        
        # Generate introduction
        intro = self._generate_narrative_intro(threads)
        document_parts.append(intro)
        
        # Follow each thread
        for thread in threads:
            thread_content = self._generate_thread_content(thread, nodes, edges)
            document_parts.append(thread_content)
        
        # Synthesize conclusions
        conclusion = self._synthesize_narrative_conclusion(threads, nodes)
        document_parts.append(conclusion)
        
        return '\n\n'.join(document_parts)
    
    def _generate_interactive(self, context: GenerationContext,
                             nodes: List[Dict], edges: List[Dict]) -> str:
        """
        Generate document with interactive elements (questions, reflections)
        """
        document_parts = []
        node_map = {node['id']: node for node in nodes}
        
        # Opening question
        opening = self._generate_opening_question(nodes)
        document_parts.append(opening)
        
        # Process nodes with interactive elements
        for i, node in enumerate(context.remaining_nodes):
            # Present the concept
            content = self._present_concept_interactively(node, i)
            document_parts.append(content)
            
            # Add reflection or question
            if i % 3 == 2:  # Every third node
                reflection = self._generate_reflection_prompt(
                    document_parts, 
                    node, 
                    nodes
                )
                document_parts.append(reflection)
        
        # Final synthesis question
        synthesis = self._generate_synthesis_question(nodes)
        document_parts.append(synthesis)
        
        return '\n\n'.join(document_parts)
    
    # Helper methods for progressive generation
    
    def _create_document_seed(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """Create initial document seed"""
        # Find foundational or high-importance nodes
        foundational = [n for n in nodes if n.get('segment_type') == 'foundational']
        if not foundational:
            foundational = sorted(nodes, key=lambda n: n.get('importance', 0), reverse=True)[:3]
        
        seed = "# Knowledge Synthesis\n\n"
        
        # Extract key concepts
        key_concepts = []
        for node in foundational[:3]:
            concept = self._extract_key_concept(node)
            if concept:
                key_concepts.append(concept)
        
        if key_concepts:
            seed += "This document explores "
            seed += self._format_concept_list(key_concepts)
            seed += ", examining their relationships and implications.\n\n"
        
        return seed
    
    def _select_next_node(self, context: GenerationContext, 
                         node_map: Dict, graph: Dict) -> Optional[Dict]:
        """Select the next node to process based on context"""
        
        # Get unvisited neighbors of visited nodes
        candidates = set()
        for visited_id in context.visited_nodes:
            if visited_id in graph:
                for neighbor_id in graph[visited_id]:
                    if neighbor_id not in context.visited_nodes:
                        candidates.add(neighbor_id)
        
        # If no neighbors, pick from remaining high-importance nodes
        if not candidates:
            for node in context.remaining_nodes:
                if node['id'] not in context.visited_nodes:
                    return node
            return None
        
        # Score candidates based on relevance to current context
        best_node = None
        best_score = -1
        
        for node_id in candidates:
            node = node_map.get(node_id)
            if not node:
                continue
                
            score = self._score_node_relevance(node, context, graph)
            if score > best_score:
                best_score = score
                best_node = node
        
        return best_node
    
    def _generate_node_content(self, node: Dict, context: GenerationContext,
                              node_map: Dict, graph: Dict) -> str:
        """Generate content for a specific node"""
        content = node.get('content', '')
        
        # Check if node has code
        if self._contains_code(content):
            return self._format_code_node(node, context)
        
        # Check node type
        seg_type = node.get('segment_type', 'general')
        
        if seg_type == 'technical_core':
            return self._format_technical_node(node, context)
        elif seg_type == 'example' or seg_type == 'illustrative':
            return self._format_example_node(node, context)
        elif seg_type == 'foundational':
            return self._format_foundational_node(node, context)
        else:
            return self._format_general_node(node, context)
    
    def _generate_transition(self, prev_text: str, next_content: str, 
                           next_node: Dict) -> str:
        """Generate transition between nodes"""
        seg_type = next_node.get('segment_type', 'general')
        
        # Transition phrases based on content type
        if seg_type == 'example':
            return "\nTo illustrate this concept:\n"
        elif seg_type == 'technical_core':
            return "\nFrom a technical perspective:\n"
        elif seg_type == 'contrast' or 'however' in next_content.lower():
            return "\nHowever, it's important to note:\n"
        elif seg_type == 'elaboration':
            return "\nBuilding on this idea:\n"
        elif 'question' in next_content.lower():
            return "\nThis raises an important question:\n"
        else:
            # Generic transition
            return "\nFurthermore:\n"
    
    def _should_start_new_section(self, context: GenerationContext, 
                                 node: Dict) -> bool:
        """Determine if we should start a new section"""
        # Start new section if:
        # 1. Different segment type
        # 2. Significant importance jump
        # 3. Reached section size limit
        
        if node.get('segment_type') != context.current_section:
            return True
            
        if len(context.generated_text.split('\n')) > 50:  # Rough section size
            return True
            
        return False
    
    def _generate_section_header(self, context: GenerationContext, 
                               node: Dict) -> str:
        """Generate appropriate section header"""
        seg_type = node.get('segment_type', 'general')
        
        headers = {
            'technical_core': "## Technical Implementation",
            'foundational': "## Core Concepts",
            'example': "## Examples and Applications",
            'implementation': "## Implementation Details",
            'problem_solving': "## Challenges and Solutions"
        }
        
        return headers.get(seg_type, "## Further Discussion")
    
    def _generate_conclusion(self, context: GenerationContext, 
                           nodes: List[Dict]) -> str:
        """Generate simple document conclusion"""
        return "\n## Conclusion\n\nDocument processing complete.\n"
    
    # Utility methods
    
    def _build_adjacency_list(self, nodes: List[Dict], 
                             edges: List[Dict]) -> Dict[str, List[str]]:
        """Build adjacency list representation of graph"""
        graph = {node['id']: [] for node in nodes}
        
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            if source in graph and target in graph:
                graph[source].append(target)
                graph[target].append(source)  # Undirected
        
        return graph
    
    def _score_node_relevance(self, node: Dict, context: GenerationContext, 
                             graph: Dict) -> float:
        """Score how relevant a node is to current context"""
        score = 0.0
        
        # Base importance
        score += node.get('importance', 0) * 0.3
        
        # Connectivity to visited nodes
        connections_to_visited = sum(
            1 for neighbor in graph.get(node['id'], [])
            if neighbor in context.visited_nodes
        )
        score += connections_to_visited * 0.4
        
        # Type continuity
        if node.get('segment_type') == context.current_section:
            score += 0.2
        
        # Content similarity (simplified)
        if context.generated_text and node.get('content'):
            # Check for shared keywords
            context_words = set(context.generated_text.lower().split()[-100:])
            node_words = set(node['content'].lower().split()[:50])
            overlap = len(context_words & node_words)
            score += min(overlap / 10, 0.3)
        
        return score
    
    def _contains_code(self, content: str) -> bool:
        """Check if content contains code"""
        code_indicators = ['```', 'def ', 'class ', 'function ', 'import ']
        return any(indicator in content for indicator in code_indicators)
    
    def _format_code_node(self, node: Dict, context: GenerationContext) -> str:
        """Format a node containing code - just return content as-is"""
        return node.get('content', '')
    
    def _format_technical_node(self, node: Dict, context: GenerationContext) -> str:
        """Format technical content"""
        return node.get('content', '')
    
    def _format_example_node(self, node: Dict, context: GenerationContext) -> str:
        """Format example content"""
        content = node.get('content', '')
        
        formatted = "*Example:*\n"
        formatted += content
        
        return formatted
    
    def _format_foundational_node(self, node: Dict, context: GenerationContext) -> str:
        """Format foundational concept"""
        content = node.get('content', '')
        
        # Extract and emphasize key concept
        if key_concept := self._extract_key_concept(node):
            formatted = f"**{key_concept}**\n\n"
            formatted += content
        else:
            formatted = content
        
        return formatted
    
    def _format_general_node(self, node: Dict, context: GenerationContext) -> str:
        """Format general content"""
        return node.get('content', '')
    
    def _extract_key_concept(self, node: Dict) -> Optional[str]:
        """Extract key concept from node"""
        content = node.get('content', '')
        
        # Look for emphasized text
        import re
        if match := re.search(r'\*\*(.+?)\*\*', content):
            return match.group(1)
        
        # Look for first sentence
        sentences = content.split('.')
        if sentences:
            first = sentences[0].strip()
            if len(first) < 100:
                return first
        
        return None
    
    def _format_concept_list(self, concepts: List[str]) -> str:
        """Format list of concepts grammatically"""
        if len(concepts) == 1:
            return concepts[0]
        elif len(concepts) == 2:
            return f"{concepts[0]} and {concepts[1]}"
        else:
            return ", ".join(concepts[:-1]) + f", and {concepts[-1]}"
    
    def _identify_narrative_threads(self, nodes: List[Dict], 
                                   edges: List[Dict]) -> List[List[Dict]]:
        """Identify narrative threads through the graph"""
        # Simplified: group by segment type
        threads = {}
        for node in nodes:
            seg_type = node.get('segment_type', 'general')
            if seg_type not in threads:
                threads[seg_type] = []
            threads[seg_type].append(node)
        
        # Sort each thread by importance
        for thread in threads.values():
            thread.sort(key=lambda n: n.get('importance', 0), reverse=True)
        
        return list(threads.values())
    
    def _generate_narrative_intro(self, threads: List[List[Dict]]) -> str:
        """Generate narrative introduction"""
        intro = "# A Journey Through Connected Concepts\n\n"
        intro += f"This narrative explores {len(threads)} interconnected themes, "
        intro += "weaving together technical insights, foundational principles, "
        intro += "and practical applications.\n"
        
        return intro
    
    def _generate_thread_content(self, thread: List[Dict], 
                               nodes: List[Dict], edges: List[Dict]) -> str:
        """Generate content for a narrative thread"""
        if not thread:
            return ""
        
        content = []
        thread_type = thread[0].get('segment_type', 'general')
        
        # Thread introduction
        content.append(f"\n## {self._humanize_segment_type(thread_type)}\n")
        
        # Process nodes in thread
        for i, node in enumerate(thread):
            node_content = node.get('content', '')
            
            # Add narrative connectors
            if i > 0:
                connector = self._get_narrative_connector(i, len(thread))
                content.append(connector)
            
            content.append(node_content)
        
        return '\n\n'.join(content)
    
    def _humanize_segment_type(self, seg_type: str) -> str:
        """Convert segment type to human-readable form"""
        type_names = {
            'technical_core': 'Technical Deep Dive',
            'foundational': 'Foundational Principles',
            'example': 'Practical Examples',
            'implementation': 'Implementation Journey',
            'problem_solving': 'Challenges and Solutions'
        }
        return type_names.get(seg_type, seg_type.replace('_', ' ').title())
    
    def _get_narrative_connector(self, position: int, total: int) -> str:
        """Get appropriate narrative connector"""
        if position == 1:
            return "Let's begin by examining..."
        elif position == total - 1:
            return "Finally, we arrive at..."
        elif position % 2 == 0:
            return "Continuing our exploration..."
        else:
            return "This naturally leads us to..."
    
    def _synthesize_narrative_conclusion(self, threads: List[List[Dict]], 
                                       nodes: List[Dict]) -> str:
        """Synthesize narrative conclusion"""
        conclusion = "\n## Synthesis\n\n"
        conclusion += "Our journey through these concepts reveals several key insights:\n\n"
        
        # Extract insights from threads
        for thread in threads:
            if thread:
                insight = self._extract_thread_insight(thread)
                conclusion += f"- {insight}\n"
        
        conclusion += "\nThese threads weave together to form a comprehensive understanding "
        conclusion += "of the subject matter, each contributing essential perspectives "
        conclusion += "to the whole."
        
        return conclusion
    
    def _extract_thread_insight(self, thread: List[Dict]) -> str:
        """Extract key insight from a thread"""
        # Use highest importance node
        if thread:
            key_node = thread[0]
            content = key_node.get('content', '')
            # Extract first meaningful sentence
            sentences = content.split('.')
            for sent in sentences:
                if len(sent.strip()) > 20:
                    return sent.strip()
        
        return "Important conceptual development"
    
    # Additional helper methods for other strategies...
    
    def _summarize_graph_content(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """Summarize the graph content"""
        summary = f"This document synthesizes {len(nodes)} interconnected concepts "
        summary += f"with {len(edges)} relationships. "
        
        # Count node types
        type_counts = {}
        for node in nodes:
            seg_type = node.get('segment_type', 'general')
            type_counts[seg_type] = type_counts.get(seg_type, 0) + 1
        
        if type_counts:
            summary += "The content includes "
            type_parts = []
            for seg_type, count in sorted(type_counts.items(), 
                                         key=lambda x: x[1], reverse=True):
                type_parts.append(f"{count} {self._humanize_segment_type(seg_type).lower()}")
            summary += ", ".join(type_parts) + "."
        
        return summary