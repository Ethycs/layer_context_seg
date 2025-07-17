#!/usr/bin/env python3
"""
Master Processor Client
=======================
This script now acts as a client to the FastAPI model server.
It sends requests to the server for all model-related tasks.
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
import time
import torch

# Add src directory to path
project_root = Path(__file__).parent.resolve()
src_path = project_root / "layered-context-graph" / "src"
sys.path.insert(0, str(src_path))

# Import config
from master_config import get_config

# Import core modules
from models.qwq_model import QwQModel
from graph.graph_reassembler import GraphReassembler
from graph.processor import GraphProcessor
from rich_pipeline import run_rich_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FullMasterProcessor:
    """A simplified processor that runs the single, unified rich pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = self.config['paths']['results_dir']
        self.qwq_model = None
        
        self.output_dir.mkdir(exist_ok=True)
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all processing components."""
        logger.info("Initializing components...")
        
        # Load the QwQModel directly
        model_path = self.config['paths']['model_path']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.qwq_model = QwQModel(model_path, device)
        
        # Initialize other components (they might not be needed for the pure LLM pipeline)
        self.graph_processor = GraphProcessor(attention_extractor=self.qwq_model)
        self.graph_reassembler = GraphReassembler()
        
        logger.info("All components initialized.")
    
    async def process_text(self, text: str, k_rules: List[str], g_rule: str) -> Dict[str, Any]:
        """Process text using the unified rich pipeline."""
        start_time = time.time()
        
        results = await run_rich_pipeline(
            text=text,
            qwq_model=self.qwq_model,
            k_rules=k_rules,
            g_rule=g_rule,
            graph_processor=self.graph_processor,
            graph_reassembler=self.graph_reassembler
        )
        
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> Path:
        """Save processing results."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qwq_layered_results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
        return output_path

async def main():
    """Main entry point for the unified rich pipeline."""
    parser = argparse.ArgumentParser(
        description='Layered Context Graph Processor',
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input text file path')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--k-rules', nargs='+', help='List of K-rules for segmentation rounds.', required=True)
    parser.add_argument('--g-rule', help='G-rule for reassembly.', required=True)
    
    args = parser.parse_args()
    
    config = get_config()
    
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
        logger.info("Starting unified rich processing...")
        results = await processor.process_text(text, args.k_rules, args.g_rule)
        output_path = processor.save_results(results)
        
        print("\n" + "="*60)
        print("UNIFIED RICH PIPELINE PROCESSING COMPLETE")
        print("="*60)
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
