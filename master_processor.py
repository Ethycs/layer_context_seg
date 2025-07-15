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
from models.context_window import ContextWindow
from models.attention_extractor import EnhancedAttentionExtractor
from models.instruction_seeder import InstructionSeeder
from partitioning.partition_manager import PartitionManager
from graph.graph_reassembler import GraphReassembler
from graph.processor import GraphProcessor
from synthesis.som_generator import SOM_DocumentGenerator
from models.qwq_model import QwQModel

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
    """Full processor with all features including QwQ integration - Fixed for full text"""
    
    def __init__(self, config: Dict[str, Any], graph_config: GraphConfig = None, qwq_model=None):
        self.config = config
        self.mode = self.config['mode']
        self.model_type = self.config['model_type']
        self.model_config = self.config['model_config']
        self.processing_settings = self.config['processing_settings']
        self.output_dir = self.config['paths']['results_dir']
        
        # The graph_config is now part of the main config
        self.graph_config = self.config.get('graph_config', GraphConfig())
            
        # Set up GPU device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components, using the pre-loaded model if provided
        self._initialize_components(qwq_model)
    
    def _initialize_components(self, qwq_model=None):
        """Initialize all processing components with a single, unified model."""
        logger.info(f"Initializing components for {self.mode} mode")

        # Use the provided qwq_model or create a new one
        if qwq_model:
            self.qwq_model = qwq_model
        else:
            self.qwq_model = QwQModel(
                model_path=self.config['model_config']['gguf_path'],
                device=self.device
            )
        
        # The EnhancedAttentionExtractor will now use the unified model
        self.attention_extractor = EnhancedAttentionExtractor(
            model_path=str(self.config['model_config']['gguf_path']),
            ollama_extractor=self.qwq_model # Pass the unified model
        )
        
        self.seeder = InstructionSeeder()
        
        self.graph_processor = GraphProcessor(
            attention_extractor=self.attention_extractor,
            ollama_extractor=self.qwq_model # Pass the unified model
        )
        self.graph_reassembler = GraphReassembler()
        self.partitioner = PartitionManager(self.attention_extractor)

        # Initialize SOM Generator
        self.som_generator = SOM_DocumentGenerator(
            llm_client=self.qwq_model, # Pass the unified model
            embedding_model=self.graph_processor.embedding_model,
            partitioner=self.partitioner
        )
        logger.info("Initialized core components for all pipelines.")
    
    def process_text(self, text: str, rules: Optional[Dict[str, str]] = None, rich: bool = False, som: bool = False) -> Dict[str, Any]:
        """Process text using the configured mode"""
        if som:
            return self._process_som_pipeline(text)
        elif rich:
            return self._process_rich_pipeline(text, rules)
        else: # Default to single-pass
            return self._process_single_pass(text, rules)
    
    def _process_som_pipeline(self, text: str) -> Dict[str, Any]:
        """
        Process text using the Self-Organizing Map pipeline.
        """
        logger.info("Starting SOM-based processing pipeline...")
        start_time = time.time()

        reassembled_text = self.som_generator.assemble_document(text)
        
        processing_time = time.time() - start_time

        return {
            'mode': 'som-pipeline',
            'input_length': len(text),
            'processing_time': processing_time,
            'output': {'reassembled_text': reassembled_text},
            'metadata': {
                'model': 'QwQ-32B + MiniLM-L6-v2',
                'architecture': 'Tape-Map-Path-Tape',
                'timestamp': datetime.now().isoformat()
            }
        }

    def _process_rich_pipeline(self, text: str, rules: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process text using the rich pipeline, which includes multi-round
        annotation and optional language guidance.
        """
        logger.info("Starting rich processing pipeline...")
        if rules:
            logger.info(f"Applying language guidance rules: {rules}")
        
        start_time = time.time()

        # This pipeline will use the multi-round processing logic
        logger.info("Phase 1: Partitioning text into segments...")
        segment_contents = self.partitioner.create_partitions(text)
        segments = [{'content': content} for content in segment_contents]
        logger.info(f"Created {len(segments)} optimal segments.")

        logger.info("Phase 2: Processing graph with multi-round annotations...")
        graph_data = self.graph_processor.process(segments, multi_round=True)
        logger.info(f"Processed graph with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges.")

        logger.info("Phase 3: Reassembling document from graph...")
        reassembled = self.graph_reassembler.reassemble(
            graph_data['nodes'],
            graph_data['edges'],
            strategy="layered_assembly",
            original_document=text
        )
        
        processing_time = time.time() - start_time

        return {
            'mode': 'rich-pipeline',
            'input_length': len(text),
            'nodes': len(graph_data['nodes']),
            'edges': len(graph_data['edges']),
            'processing_time': processing_time,
            'output': reassembled,
            'metadata': {
                'model': 'QwQ-32B',
                'architecture': 'Tape-to-Graph-Rich',
                'timestamp': datetime.now().isoformat()
            }
        }

    def _process_single_pass(self, text: str, rules: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process text using the simplified, three-phase pipeline.
        """
        logger.info("Starting refactored single-pass processing...")
        start_time = time.time()

        # 1. Disassembly Phase (Tape to Nodes)
        logger.info("Phase 1: Partitioning text into segments...")
        segment_contents = self.partitioner.create_partitions(text)
        logger.info(f"Created {len(segment_contents)} optimal segments.")

        # Prepare segments for graph processor, which expects a list of dicts
        segments = [{'content': content} for content in segment_contents]

        # 2. Reconstruction Phase (Nodes to Graph)
        logger.info("Phase 2: Processing segments into a graph...")
        graph_data = self.graph_processor.process(segments)
        logger.info(f"Processed graph with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges.")

        # 3. Reassembly Phase (Graph to Document)
        logger.info("Phase 3: Reassembling document from graph...")
        reassembled = self.graph_reassembler.reassemble(
            graph_data['nodes'],
            graph_data['edges'],
            strategy="layered_assembly",
            original_document=text
        )
        
        processing_time = time.time() - start_time

        return {
            'mode': 'refactored-single-pass',
            'input_length': len(text),
            'nodes': len(graph_data['nodes']),
            'edges': len(graph_data['edges']),
            'processing_time': processing_time,
            'output': reassembled,
            'metadata': {
                'model': 'QwQ-32B',
                'architecture': 'Tape-to-Graph-Refactored',
                'timestamp': datetime.now().isoformat()
            }
        }

    def _classify_edge_relationships(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """Use LLM to classify the relationship type for important edges."""
        if not self.ollama_extractor:
            logger.warning("LLM extractor not available, skipping edge classification.")
            return edges

        node_map = {node['id']: node for node in nodes}
        
        important_nodes = {n['id'] for n in nodes if n.get('classification') in ['KEEP', 'TRACK']}
        
        for edge in tqdm(edges, desc="Classifying Edges"):
            source_id = edge.get('source')
            target_id = edge.get('target')

            if source_id in important_nodes and target_id in important_nodes:
                source_node = node_map.get(source_id)
                target_node = node_map.get(target_id)

                if source_node and target_node:
                    if not hasattr(self, 'llm_synthesizer'):
                        from synthesis.llm_tape_synthesizer import LLMTapeSynthesizer
                        self.llm_synthesizer = LLMTapeSynthesizer(self.config['paths']['project_root'] / 'qwq.gguf')

                    relationship = self.llm_synthesizer.classify_edge_relationship(
                        source_node['content'],
                        target_node['content']
                    )
                    edge['type'] = relationship
                    edge['classified'] = True
        
        return edges
    
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
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Full Layered Context Graph Processor with QwQ Integration',
    )
    
    parser.add_argument(
        '--rich',
        action='store_true',
        help='Enable rich processing with multi-round annotation and language guidance.'
    )
    parser.add_argument(
        '--som',
        action='store_true',
        help='Enable the Self-Organizing Map pipeline.'
    )
    
    parser.add_argument('--input', '-i', help='Input text file path')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--demo', choices=list(DEMO_CONFIGS.keys()) + ['layered_context_file'],
                       help='Use demo content')
    
    args = parser.parse_args()
    
    # Determine mode based on flags
    if args.som:
        mode = 'som-pipeline'
    elif args.rich:
        mode = 'rich'
    else:
        mode = 'single-pass'
    config = get_config(mode=mode, model_type='ollama')
    
    if args.output:
        config['paths']['results_dir'] = Path(args.output)
    
    from load_demo_content import get_demo_content
    if args.demo:
        text = get_demo_content(args.demo)
        logger.info(f"Using demo content: {args.demo}")
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"Loaded input file: {input_path} ({len(text)} characters)")
    else:
        logger.error("Either --input or --demo must be specified")
        sys.exit(1)
    
    try:
        # The model is now loaded directly by the FullMasterProcessor
        processor = FullMasterProcessor(config)
        logger.info(f"Starting {mode} processing with pre-loaded QwQ...")
        results = processor.process_text(text, rich=args.rich, som=args.som)
        output_path = processor.save_results(results)
        
        print("\n" + "="*60)
        print("QWQ-POWERED PROCESSING COMPLETE")
        print("="*60)
        print(f"Mode: {results['mode']}")
        print(f"Model: {results['metadata'].get('model', 'QwQ-32B')}")
        print(f"Input length: {results.get('input_length', len(text))} characters")
        print(f"Knowledge graph nodes: {results['nodes']}")
        print(f"Knowledge graph edges: {results['edges']}")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
