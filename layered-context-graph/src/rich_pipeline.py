#!/usr/bin/env python3
"""
Unified Rich Processing Pipeline
================================
This module defines the single, "as-rich-as-possible" pipeline for the
Tape -> Tree -> Graph transformation. It orchestrates the Disassembly and
Reassembly phases to produce a high-quality, structured output.
"""

import logging
from typing import Dict, Any, List
import json

from partitioning.partition_manager import PartitionManager
from synthesis.graph_reassembler import GraphReassembler
from models.qwq_model import QwQModel
from models.baai_model import BAAIModel

logger = logging.getLogger(__name__)

async def run_rich_pipeline(
    text: str,
    qwq_model: QwQModel,
    k_rules: List[str],
    g_rule: str,
    graph_processor=None,  # Deprecated parameter for compatibility
    graph_reassembler: GraphReassembler = None
) -> Dict[str, Any]:
    """
    Executes the full, unified, rich processing pipeline.

    Args:
        text: The raw input text.
        qwq_model: An initialized QwQModel instance for segmentation and analysis.
        k_rules: A list of natural language rules for segmentation.
        g_rule: A natural language rule for reassembly.
        graph_processor: Deprecated, kept for compatibility.
        graph_reassembler: Optional GraphReassembler instance.

    Returns:
        A dictionary containing the final processing results.
    """
    logger.info("--- Starting Unified Rich Pipeline ---")

    # Initialize BAAI model for embeddings
    from pathlib import Path
    baai_model_path = Path('./bge-en-icl')  # Default path, should be configurable
    baai_model = BAAIModel(str(baai_model_path))
    
    # 1. Use QwQ for attention-based segmentation if k_rules request it
    use_attention_segmentation = any('attention' in rule.lower() for rule in k_rules)
    
    if use_attention_segmentation:
        logger.info("Phase 1a: Using attention-based segmentation...")
        segments = qwq_model.segment_by_attention(text, use_low_rank=True)
        # Convert segments to dict format expected by PartitionManager
        segment_dicts = []
        current_pos = 0
        for seg_text in segments:
            segment_dicts.append({'content': seg_text, 'start': current_pos})
            current_pos += len(seg_text)
    else:
        segment_dicts = None
    
    # 2. Create partition graph
    logger.info("Phase 1b: Creating partition graph...")
    partition_manager = PartitionManager(embedding_model=baai_model)
    
    if segment_dicts:
        # Use pre-segmented text
        graph = partition_manager._build_graph(segment_dicts)
        nodes, edges = partition_manager._build_hierarchy(graph)
        nodes = partition_manager._classify_nodes(nodes, edges)
        graph_data = {'nodes': nodes, 'edges': edges}
    else:
        # Use default segmentation
        graph_data = partition_manager.create_partition_graph(text)
    
    logger.info(f"Graph construction complete. Produced {len(graph_data['nodes'])} nodes.")

    # 3. Reassembly Phase
    logger.info("Phase 2: Reassembling document...")
    if graph_reassembler is None:
        graph_reassembler = GraphReassembler()
    final_output = graph_reassembler.reassemble(
        graph_data['nodes'],
        graph_data['edges'],
        strategy=g_rule
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
