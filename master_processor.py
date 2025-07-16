#!/usr/bin/env python3
"""
Full Master Processor for Layered Context Graph with QwQ Integration - FIXED
============================================================================
This version fixes text truncation issues and ensures full content preservation.
"""

import sys
import os
import re
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time

# Add src directory to path
project_root = Path(__file__).parent.resolve()
src_path = project_root / "layered-context-graph" / "src"
sys.path.insert(0, str(src_path))

# Import config
from master_config import get_config, get_rule_set, DEMO_CONFIGS, RULE_SETS, GraphConfig

# Import core modules
from models.attention_extractor import EnhancedAttentionExtractor
from partitioning.partition_manager import PartitionManager
from graph.graph_reassembler import GraphReassembler
from graph.processor import GraphProcessor
from models.qwq_model import QwQModel
from rich_pipeline import run_rich_pipeline

# Try to import TorchSpectralProcessor for hybrid processing
try:
    sys.path.insert(0, str(project_root))  # Add project root for torch_spectral_processor
    from torch_spectral_processor import TorchSpectralProcessor
    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False
    TorchSpectralProcessor = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FullMasterProcessor:
    """A simplified processor that runs the single, unified rich pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = self.config['paths']['results_dir']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        self.output_dir.mkdir(exist_ok=True)
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all processing components."""
        logger.info("Initializing components for the unified rich pipeline...")

        self.qwq_model = QwQModel(
            model_path=self.config['model_config']['gguf_path'],
            device=self.device
        )
        
        self.attention_extractor = EnhancedAttentionExtractor(qwq_model=self.qwq_model)
        self.partitioner = PartitionManager(self.attention_extractor)
        self.graph_processor = GraphProcessor(
            attention_extractor=self.attention_extractor,
            ollama_extractor=self.qwq_model
        )
        self.graph_reassembler = GraphReassembler()
        
        logger.info("All components initialized.")
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text using the unified rich pipeline."""
        start_time = time.time()
        
        results = run_rich_pipeline(
            text=text,
            partition_manager=self.partitioner,
            graph_processor=self.graph_processor,
            graph_reassembler=self.graph_reassembler
        )
        
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> Path:
        """Save processing results with full text preservation"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qwq_layered_results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        def custom_serializer(obj):
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {k: custom_serializer(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [custom_serializer(item) for item in obj]
            else:
                return str(obj)
        
        serialized_results = custom_serializer(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serialized_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
        
        text_path = output_path.with_suffix('.txt')
        self._save_text_results(results, text_path)
        
        return output_path
    
    def _save_text_results(self, results: Dict[str, Any], output_path: Path):
        """Save human-readable text results with node list"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("LAYERED CONTEXT GRAPH PROCESSING RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Processing Mode: {results.get('mode', 'N/A')}\n")
            f.write(f"Input Length: {results.get('input_length', 0):,} characters\n")
            f.write(f"Nodes Created: {results.get('nodes', 0)}\n")
            f.write(f"Edges Created: {results.get('edges', 0)}\n")
            f.write(f"Processing Time: {results.get('processing_time', 0):.2f} seconds\n")
            
            if 'metadata' in results:
                meta = results['metadata']
                f.write(f"Model: {meta.get('model', 'N/A')}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            
            if 'output' in results and 'nodes' in results['output']:
                nodes = results['output']['nodes']
                f.write(f"\nNODE LIST ({len(nodes)} nodes)\n")
                f.write("-" * 60 + "\n\n")
                
                for i, node in enumerate(tqdm(nodes, desc="Saving Nodes")):
                    f.write(f"Node {i} [{node.get('id', f'node_{i}')}]\n")
                    content = node.get('content', '')
                    f.write(f"Content Length: {len(content)} chars\n")
                    if 'classification' in node:
                        f.write(f"Classification: {node['classification']}\n")
                    if 'importance_score' in node:
                        f.write(f"Importance: {node['importance_score']:.3f}\n")
                    f.write("\n")
            
            f.write("=" * 60 + "\n")
            f.write("RECONSTRUCTED DOCUMENT\n")
            f.write("=" * 60 + "\n\n")
            
            if 'output' in results and 'reassembled_text' in results['output']:
                f.write(results['output']['reassembled_text'])
            else:
                f.write("[No reconstructed document available]\n")
        
        logger.info(f"Text results saved to {output_path}")

def main():
    """Main entry point for the unified rich pipeline."""
    parser = argparse.ArgumentParser(
        description='Layered Context Graph Processor with a Unified Rich Pipeline',
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input text file path')
    parser.add_argument('--output', '-o', help='Output directory')
    
    args = parser.parse_args()
    
    config = get_config(model_type='ollama')
    
    if args.output:
        config['paths']['results_dir'] = Path(args.output)
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    logger.info(f"Loaded input file: {input_path} ({len(text)} characters)")
    
    try:
        processor = FullMasterProcessor(config)
        logger.info("Starting unified rich processing with pre-loaded QwQ...")
        results = processor.process_text(text)
        output_path = processor.save_results(results)
        
        print("\n" + "="*60)
        print("UNIFIED RICH PIPELINE PROCESSING COMPLETE")
        print("="*60)
        print(f"Input length: {results.get('input_length', len(text))} characters")
        print(f"Knowledge graph nodes: {results.get('nodes', 0)}")
        print(f"Knowledge graph edges: {results.get('edges', 0)}")
        print(f"Processing time: {results.get('processing_time', 0):.2f} seconds")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
