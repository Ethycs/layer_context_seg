import networkx as nx
import re
from typing import Dict, List, Any

class GraphReassembler:
    """
    A consolidated synthesizer that transforms a knowledge graph back into
    various human-readable document formats (Tape₂).
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.strategies = {
            'executive_summary': self._synthesize_executive_summary,
            'tutorial': self._synthesize_tutorial,
            'reference': self._synthesize_reference,
            'readme': self._synthesize_readme
        }

    def reassemble(self, nodes: List[Dict], edges: List[Dict], strategy: str) -> str:
        """
        Main entry point for reassembly. Routes to the specified strategy.
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown synthesis strategy: {strategy}")
        
        # Filter for nodes that are not marked for deletion
        active_nodes = [n for n in nodes if n.get('tag') != 'DELETE']
        
        return self.strategies[strategy](active_nodes, edges)

    def _synthesize_executive_summary(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """Generates a concise executive summary from the most important nodes."""
        important_nodes = sorted(nodes, key=lambda n: n.get('importance', 0), reverse=True)
        
        summary_parts = ["# Executive Summary\n"]
        summary_parts.append(f"This document summarizes {len(nodes)} key concepts, focusing on the most critical insights.\n")
        
        for node in important_nodes[:5]: # Top 5 nodes for summary
            summary_parts.append(f"## {node.get('id')}\n{node.get('content')}\n")
            
        return "\n".join(summary_parts)

    def _synthesize_tutorial(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """Generates a step-by-step tutorial from the graph."""
        # A real implementation would use topological sort on the graph
        # For now, we sort by start position to maintain a logical flow
        sorted_nodes = sorted(nodes, key=lambda n: n.get('start_pos', 0))
        
        tutorial_parts = ["# Tutorial\n"]
        for i, node in enumerate(sorted_nodes):
            tutorial_parts.append(f"## Step {i+1}: {node.get('id')}\n")
            tutorial_parts.append(node.get('content'))
            if node.get('has_code'):
                tutorial_parts.append("\n*Example Code Block Included*")
        
        return "\n\n".join(tutorial_parts)

    def _synthesize_reference(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """Generates a reference document."""
        # Group nodes by type for a structured reference
        grouped_nodes = {}
        for node in nodes:
            node_type = 'Code' if node.get('has_code') else 'Math' if node.get('has_math') else 'Text'
            if node_type not in grouped_nodes:
                grouped_nodes[node_type] = []
            grouped_nodes[node_type].append(node)
            
        ref_parts = ["# Reference Guide\n"]
        for group_name, group_nodes in grouped_nodes.items():
            ref_parts.append(f"## {group_name} Segments\n")
            for node in group_nodes:
                ref_parts.append(f"### ID: {node['id']}\n{node['content']}\n")
        
        return "\n".join(ref_parts)

    def _synthesize_readme(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """Generates a README.md style document."""
        readme_parts = ["# Project README\n"]
        
        # Find a good title/intro node
        intro_node = min(nodes, key=lambda n: n.get('start_pos', 0))
        readme_parts.append(f"## Overview\n{intro_node['content']}\n")
        
        # Add code blocks as feature examples
        code_nodes = [n for n in nodes if n.get('has_code')]
        if code_nodes:
            readme_parts.append("## Features & Examples\n")
            for node in code_nodes:
                readme_parts.append(f"### Feature: {node['id']}\n{node['content']}\n")
                
        return "\n".join(readme_parts)
