#!/usr/bin/env python3
"""
Knowledge Graph Manager
=======================
This module provides a manager for classifying nodes in a knowledge graph
based on their structural importance and content.
"""

import networkx as nx
from typing import Dict, List, Any
import logging
from sklearn.cluster import AgglomerativeClustering
import numpy as np

logger = logging.getLogger(__name__)

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

    def condense_graph(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], llm_extractor, embedding_model) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
        """
        Main entry point to condense the graph.
        """
        logger.info(f"Starting graph condensation on {len(nodes)} nodes.")
        
        text_nodes, code_nodes, other_nodes = self._classify_node_types(nodes)
        
        condensed_text_nodes, text_mappings = self._condense_text_nodes(text_nodes, llm_extractor, embedding_model)
        
        deduplicated_code_nodes, code_deletions = self._deduplicate_code_nodes(code_nodes)
        
        final_nodes = condensed_text_nodes + deduplicated_code_nodes + other_nodes
        
        all_mappings = {**text_mappings, **code_deletions}
        final_edges = self._rewire_edges(edges, all_mappings)
        
        logger.info(f"Graph condensation complete. New node count: {len(final_nodes)}")
        return final_nodes, final_edges

    def _classify_node_types(self, nodes: List[Dict[str, Any]]) -> (List[Dict], List[Dict], List[Dict]):
        """Classify nodes into text, code, and other categories."""
        text_nodes, code_nodes, other_nodes = [], [], []
        for node in nodes:
            if '```' in node.get('content', ''):
                code_nodes.append(node)
            else:
                text_nodes.append(node)
        return text_nodes, code_nodes, other_nodes

    def _condense_text_nodes(self, nodes: List[Dict[str, Any]], llm_extractor, embedding_model) -> (List[Dict], Dict):
        """Merge clusters of similar text nodes using an LLM."""
        if len(nodes) < 2 or not embedding_model:
            return nodes, {}
            
        logger.info(f"Condensing {len(nodes)} text nodes.")
        
        contents = [node['content'] for node in nodes]
        embeddings = embedding_model.encode(contents, show_progress_bar=False)
        
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.7, linkage='average')
        labels = clustering.fit_predict(embeddings)
        
        new_nodes = []
        mappings = {}
        
        for cluster_id in set(labels):
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
            
            if len(cluster_indices) == 1:
                new_nodes.append(nodes[cluster_indices[0]])
                continue
            
            cluster_nodes = [nodes[i] for i in cluster_indices]
            logger.info(f"Merging cluster {cluster_id} with {len(cluster_nodes)} nodes.")
            
            new_content = self._synthesize_content(cluster_nodes, llm_extractor)
            
            new_node_id = f"condensed_text_{cluster_id}"
            new_node = {
                'id': new_node_id,
                'content': new_content,
                'level': min(n.get('level', 0) for n in cluster_nodes),
                'classification': 'KEEP',
                'is_condensed': True
            }
            new_nodes.append(new_node)
            
            for old_node in cluster_nodes:
                mappings[old_node['id']] = new_node_id
                
        return new_nodes, mappings

    def _synthesize_content(self, nodes_to_merge: List[Dict[str, Any]], llm_extractor) -> str:
        """Use LLM to synthesize content from a list of nodes."""
        if not llm_extractor:
            return "\n\n".join([n['content'] for n in nodes_to_merge])
            
        prompt = "The following text segments are very similar. Please synthesize them into a single, coherent, and concise paragraph that captures the core idea from all of them. Do not lose any key information.\n\n"
        for i, node in enumerate(nodes_to_merge):
            prompt += f"--- Segment {i+1} ---\n{node['content']}\n\n"
        prompt += "--- Synthesized Paragraph ---\n"
        
        try:
            response = llm_extractor.generate_text(prompt, max_length=512)
            return response.strip()
        except Exception as e:
            logger.error(f"LLM synthesis for condensing failed: {e}")
            return "\n\n".join([n['content'] for n in nodes_to_merge])

    def _deduplicate_code_nodes(self, nodes: List[Dict[str, Any]]) -> (List[Dict], Dict):
        """Find the 'most definitive' version of similar code blocks."""
        logger.info(f"De-duplicating {len(nodes)} code nodes (placeholder).")
        return nodes, {}

    def _rewire_edges(self, edges: List[Dict[str, Any]], mappings: Dict) -> List[Dict[str, Any]]:
        """Update edges to point to new, condensed nodes."""
        new_edges = []
        seen_edges = set()
        for edge in edges:
            source = mappings.get(edge['source'], edge['source'])
            target = mappings.get(edge['target'], edge['target'])
            
            if source == target:
                continue
            
            edge_tuple = tuple(sorted((source, target)))
            if edge_tuple in seen_edges:
                continue
            
            edge['source'] = source
            edge['target'] = target
            new_edges.append(edge)
            seen_edges.add(edge_tuple)
            
        return new_edges
