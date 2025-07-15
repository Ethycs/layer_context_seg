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

    def __init__(self, attention_extractor=None, ollama_extractor=None):
        """
        Initialize the GraphProcessor with necessary components.
        
        Args:
            attention_extractor: Used for attention-based splitting and edge detection.
                                 Relies on the primary 'qwq' model.
            ollama_extractor: The primary LLM ('qwq') used for high-level reasoning
                              and content synthesis.
        """
        self.attention_graph_builder = AttentionGraphBuilder(attention_extractor=attention_extractor)
        self.hierarchical_builder = HierarchicalGraphBuilder()
        # The ollama_extractor is the main reasoning engine (e.g., qwq model)
        self.ollama_extractor = ollama_extractor
        # The embedding_model is a specialized, lightweight model for calculating similarity
        self.embedding_model = self._load_embedding_model()

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

    def process(self, segments: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        Execute the full graph processing pipeline.

        Args:
            segments: A list of optimal segments from the partitioning phase.

        Returns:
            A dictionary containing the final hierarchical nodes and edges.
        """
        logger.info(f"Starting graph processing for {len(segments)} segments.")

        # 1. Build Initial Graph from Segments
        # Note: The current AttentionGraphBuilder expects a list of strings.
        # We adapt by extracting the content from the segments.
        segment_contents = [s['content'] for s in segments]
        initial_graph = self.attention_graph_builder.build_from_attention({}, segment_contents)
        
        nodes = initial_graph.get('nodes', [])
        edges = initial_graph.get('edges', [])

        # 2. Classify and Condense Graph
        if not nodes:
            logger.warning("No nodes were generated from the segments. Aborting graph processing.")
            return {'nodes': [], 'edges': []}

        kg_manager = KnowledgeGraphManager({"nodes": nodes, "edges": edges})
        
        logger.info("Classifying nodes...")
        classified_nodes = kg_manager.classify_nodes()
        
        logger.info("Condensing graph...")
        condensed_nodes, condensed_edges = kg_manager.condense_graph(
            classified_nodes, edges, self.ollama_extractor, self.embedding_model
        )

        # 3. Build Hierarchical Structure
        logger.info("Building final hierarchical graph...")
        hierarchical_nodes, tree_edges = self.hierarchical_builder.build_hierarchy(
            condensed_nodes, condensed_edges
        )

        logger.info("Graph processing complete.")
        return {
            "nodes": hierarchical_nodes,
            "edges": tree_edges
        }
