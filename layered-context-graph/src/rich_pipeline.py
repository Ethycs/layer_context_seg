#!/usr/bin/env python3
"""
Unified Rich Processing Pipeline
================================
This module defines the single, "as-rich-as-possible" pipeline for the
Tape -> Tree -> Graph transformation. It orchestrates the Disassembly and
Reassembly phases to produce a high-quality, structured output.
"""

import logging
from typing import Dict, Any

from partitioning.partition_manager import PartitionManager
from graph.processor import GraphProcessor
from graph.graph_reassembler import GraphReassembler

logger = logging.getLogger(__name__)

def run_rich_pipeline(
    text: str,
    partition_manager: PartitionManager,
    graph_processor: GraphProcessor,
    graph_reassembler: GraphReassembler
) -> Dict[str, Any]:
    """
    Executes the full, unified, rich processing pipeline.

    Args:
        text: The raw input text.
        partition_manager: An initialized PartitionManager instance.
        graph_processor: An initialized GraphProcessor instance.
        graph_reassembler: An initialized GraphReassembler instance.

    Returns:
        A dictionary containing the final processing results.
    """
    logger.info("--- Starting Unified Rich Pipeline ---")

    # 1. Disassembly Phase (Tape -> Segments)
    logger.info("Phase 1: Disassembly - Creating optimal segments...")
    optimal_segments = partition_manager.create_partitions(text)
    logger.info(f"Disassembly complete. Produced {len(optimal_segments)} optimal segments.")

    # 2. Reassembly Phase (Segments -> Graph -> Output)
    logger.info("Phase 2: Reassembly - Building and enriching the graph...")
    
    # The GraphProcessor will be enhanced to perform multi-round enrichment
    graph_data = graph_processor.process(
        segments=[{'content': s} for s in optimal_segments],
        multi_round=True # This will be implemented next
    )
    logger.info(f"Graph processing complete. Final graph has {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges.")

    # Reassemble the final document from the enriched graph
    final_output = graph_reassembler.reassemble(
        graph_data['nodes'],
        graph_data['edges'],
        strategy="layered_assembly",
        original_document=text
    )
    logger.info("Reassembly complete. Final output generated.")
    
    logger.info("--- Unified Rich Pipeline Finished ---")

    return {
        'input_length': len(text),
        'nodes': len(graph_data['nodes']),
        'edges': len(graph_data['edges']),
        'output': final_output,
        'metadata': {
            'pipeline': 'unified_rich_pipeline',
            'architecture': 'Tape-Tree-Graph'
        }
    }
