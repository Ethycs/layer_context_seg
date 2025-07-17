#!/usr/bin/env python3
"""
Unified Graph Processor
=======================
This module consolidates the logic for graph creation, manipulation, and
hierarchical structuring into a single, cohesive processor.
"""

import logging
from typing import Dict, List, Any

from .attention_graph_builder import AttentionGraphBuilder
from .knowledge_graph_manager import KnowledgeGraphManager
from .hierarchical_graph_builder import HierarchicalGraphBuilder

logger = logging.getLogger(__name__)

class GraphProcessor:
    """
    Orchestrates the entire graph processing pipeline, from segment-based
    graph creation to hierarchical structuring.
    """

    def __init__(self, attention_extractor=None, ollama_extractor=None, config=None):
        """
        Initialize the GraphProcessor with necessary components.
        
        Args:
            attention_extractor: Used for attention-based splitting and edge detection.
                                 Relies on the primary 'qwq' model.
            ollama_extractor: The primary LLM ('qwq') used for high-level reasoning
                              and content synthesis.
            config: The main application configuration.
        """
        self.attention_graph_builder = AttentionGraphBuilder(attention_extractor=attention_extractor)
        self.hierarchical_builder = HierarchicalGraphBuilder()
        self.attention_extractor = attention_extractor
        self.ollama_extractor = ollama_extractor
        self.embedding_model = self._load_embedding_model()
        self.config = config or {}

    def _load_embedding_model(self):
        """
        Loads the SentenceTransformer embedding model.
        This model is used for efficient semantic similarity calculations, which is a
        different task than the complex reasoning handled by the main ollama_extractor.
        """
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            logger.warning("SentenceTransformers not installed. Text node condensation will be skipped.")
            return None

    async def process(self, segments: List[Any], multi_round: bool = False, rounds: int = 3) -> Dict[str, List[Dict]]:
        """
        Execute the full graph processing pipeline, with optional multi-round enrichment.
        """
        logger.info(f"Starting graph processing for {len(segments)} segments.")
        
        # Initialize the KnowledgeGraphManager with the embedding model
        kg_manager = KnowledgeGraphManager(embedding_model=self.embedding_model)
        
        # Build the initial graph from enriched segments
        initial_graph = kg_manager.build_initial_graph(segments)
        nodes = initial_graph['nodes']
        edges = initial_graph['edges']

        if not nodes:
            logger.warning("No nodes were generated. Aborting graph processing.")
            return {'nodes': [], 'edges': []}

        if multi_round:
            logger.info(f"--- Starting Multi-Round Graph Enrichment ({rounds} rounds) ---")
            for i in range(rounds):
                logger.info(f"--- Enrichment Round {i+1}/{rounds} ---")
                
                logger.info("Classifying nodes based on graph structure...")
                nodes = kg_manager.classify_nodes({'nodes': nodes, 'edges': edges})
                
                nodes_to_keep = [n for n in nodes if n.get('tag') != 'DELETE']
                
                logger.info("Condensing graph by merging similar nodes...")
                nodes, edges = await kg_manager.condense_graph(nodes_to_keep, edges, self.ollama_extractor)
                
                logger.info(f"Round {i+1} complete. Nodes: {len(nodes)}, Edges: {len(edges)}")

            logger.info("--- Multi-Round Enrichment Complete ---")
        else:
            # Single-pass processing
            nodes = kg_manager.classify_nodes({'nodes': nodes, 'edges': edges})
            nodes_to_keep = [n for n in nodes if n.get('tag') != 'DELETE']
            nodes, edges = await kg_manager.condense_graph(nodes_to_keep, edges, self.ollama_extractor)

        logger.info("Building final hierarchical structure...")
        hierarchical_nodes, tree_edges = self.hierarchical_builder.build_hierarchy(nodes, edges)

        logger.info("Graph processing complete.")
        return {"nodes": hierarchical_nodes, "edges": tree_edges}
