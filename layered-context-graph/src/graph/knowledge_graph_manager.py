#!/usr/bin/env python3
"""
Knowledge Graph Manager
=======================
This module provides a manager for classifying nodes in a knowledge graph
based on their structural importance and content.
"""

import networkx as nx
from typing import Dict, List, Any

class KnowledgeGraphManager:
    """
    Manages and classifies nodes within a knowledge graph.
    """
    
    def __init__(self, graph: Dict[str, List[Dict]]):
        """
        Initialize the manager with a graph.
        
        Args:
            graph: A dictionary with 'nodes' and 'edges' keys.
        """
        self.graph_dict = graph
        self.nx_graph = self._create_networkx_graph(graph)
    
    def _create_networkx_graph(self, graph_dict: Dict[str, List[Dict]]) -> nx.DiGraph:
        """
        Convert dictionary-based graph to a NetworkX DiGraph.
        """
        G = nx.DiGraph()
        
        # Add nodes
        for node_data in graph_dict.get('nodes', []):
            node_id = node_data.get('id')
            if node_id is not None:
                G.add_node(node_id, **node_data)
        
        # Add edges
        for edge_data in graph_dict.get('edges', []):
            source = edge_data.get('source')
            target = edge_data.get('target')
            if source is not None and target is not None and G.has_node(source) and G.has_node(target):
                G.add_edge(source, target, weight=edge_data.get('weight', 1.0))
                
        return G

    def classify_nodes(self, pagerank_alpha: float = 0.85, importance_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Classify nodes as KEEP, TRACK, or DELETE based on PageRank and content.
        
        Args:
            pagerank_alpha: Damping parameter for PageRank.
            importance_threshold: Minimum PageRank score to be considered for KEEP.
            
        Returns:
            The list of node dictionaries with an added 'classification' key.
        """
        if not self.nx_graph.nodes:
            return self.graph_dict.get('nodes', [])

        # Calculate PageRank to determine node importance
        try:
            pagerank = nx.pagerank(self.nx_graph, alpha=pagerank_alpha)
        except nx.PowerIterationFailedConvergence:
            # Fallback for graphs where PageRank doesn't converge
            pagerank = {node: 1.0 / len(self.nx_graph) for node in self.nx_graph.nodes()}

        nodes = self.graph_dict.get('nodes', [])
        for node in nodes:
            node_id = node.get('id')
            if node_id not in self.nx_graph:
                node['classification'] = 'DELETE' # Should not happen if graph is built correctly
                continue

            score = pagerank.get(node_id, 0)
            content = node.get('content', '')

            # Default classification
            classification = 'DELETE'

            # Rule 1: High importance nodes are kept
            if score > importance_threshold:
                classification = 'KEEP'
            
            # Rule 2: Nodes with explicit tracking keywords are tracked
            if 'TODO' in content or 'FIXME' in content or '?' in content:
                classification = 'TRACK'

            # Rule 3: Code blocks are generally kept unless they are trivial
            if node.get('features', {}).get('has_code', False):
                if len(content.splitlines()) > 2: # Keep non-trivial code blocks
                    classification = 'KEEP'

            # Rule 4: Very short, non-essential nodes are deleted
            if len(content.split()) < 10 and classification != 'TRACK':
                 classification = 'DELETE'

            node['classification'] = classification
            node['importance_score'] = score

        return nodes
