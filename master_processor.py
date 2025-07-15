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

# Add src directory to path
project_root = Path(__file__).parent.resolve()
src_path = project_root / "layered-context-graph" / "src"
sys.path.insert(0, str(src_path))

# Import config
from master_config import get_config, get_rule_set, DEMO_CONFIGS, RULE_SETS

# Import centralized graph configuration
try:
    from config.graph_config import GraphConfig, get_config as get_graph_config
except ImportError:
    GraphConfig = None
    get_graph_config = None

# Import core modules
from models.context_window import ContextWindow
from models.attention_extractor import EnhancedAttentionExtractor
from models.instruction_seeder import InstructionSeeder
from partitioning.partition_manager import PartitionManager
from graph.graph_reassembler import GraphReassembler
from graph.attention_graph_builder import AttentionGraphBuilder
from graph.hierarchical_graph_builder import HierarchicalGraphBuilder
from graph.knowledge_graph_manager import KnowledgeGraphManager
from utils.performance_monitor import Timer, timed, gpu_memory_tracked

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
    
    def __init__(self, config: Dict[str, Any], graph_config: GraphConfig = None, ollama_extractor=None):
        self.config = config
        self.mode = self.config['mode']
        self.model_type = self.config['model_type']
        self.model_config = self.config['model_config']
        self.processing_settings = self.config['processing_settings']
        self.output_dir = self.config['paths']['results_dir']
        
        # Use centralized graph configuration or create from mode
        if graph_config:
            self.graph_config = graph_config
        elif GraphConfig and get_graph_config:
            # Get preset based on mode
            preset = 'default'
            if self.mode == 'spectral-hybrid':
                preset = 'spectral'
            elif 'conversation' in str(self.config.get('demo', '')):
                preset = 'conversation'
            self.graph_config = get_graph_config(preset)
        else:
            self.graph_config = None
            
        # Set up GPU device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components, using the pre-loaded model if provided
        self._initialize_components(ollama_extractor)
    
    @gpu_memory_tracked
    def _initialize_components(self, ollama_extractor=None):
        """Initialize all processing components with QwQ"""
        logger.info(f"Initializing components for {self.mode} mode")
        
        self.ollama_extractor = ollama_extractor
        if self.ollama_extractor is None:
            logger.info("No pre-loaded model provided, initializing new one.")
            qwq_path = self.config['paths']['project_root'] / 'qwq.gguf'
            if qwq_path.exists():
                from utils.llm_connection_pool import get_global_pool
                self.connection_pool = get_global_pool()
                self.ollama_extractor = self.connection_pool.get_connection(
                    str(qwq_path), "ollama", device=self.device
                )
            else:
                self.ollama_extractor = None
        
        from models.attention_extractor import EnhancedAttentionExtractor
        self.attention_extractor = EnhancedAttentionExtractor(
            model_path=str(self.config['paths']['project_root'] / 'qwq.gguf'),
            ollama_extractor=self.ollama_extractor
        )
        
        self.seeder = InstructionSeeder()
        
        try:
            from graph.torch_attention_graph_builder import TorchAttentionGraphBuilder
            self.torch_graph_builder = TorchAttentionGraphBuilder(device=self.device)
            logger.info("Using GPU-accelerated TorchAttentionGraphBuilder")
        except ImportError:
            self.torch_graph_builder = None
            logger.info("TorchAttentionGraphBuilder not available")
        
        self.attention_graph_builder = AttentionGraphBuilder(
            attention_extractor=self.attention_extractor
        )
        
        self.condenser = GraphCondenser(self.ollama_extractor)
        
self.graph_reassembler = GraphReassembler()
logger.info("Using unified graph reassembler with multiple strategies")
            
        self.hierarchical_builder = HierarchicalGraphBuilder()
    
    def process_text(self, text: str, rules: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process text using the configured mode"""
        return self._process_single_pass(text, rules)
    
    def _hierarchical_partition(self, text: str) -> List[Dict[str, Any]]:
        """Performs hierarchical, iterative segmentation of a text document."""
        logger.info("Starting hierarchical partitioning...")
        
        initial_segment = {'content': text, 'level': 0}
        
        segments = self._iterative_segmentation([initial_segment])
        
        logger.info(f"Hierarchical partitioning complete. Generated {len(segments)} segments.")
        return segments

    def _iterative_segmentation(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Implements iterative segmentation."""
        config = {'max_rounds': 5, 'target_segment_size': 400, 'min_segment_size': 100, 'max_segment_size': 1200}
        for i in range(config['max_rounds']):
            logger.info(f"--- Partitioning Round {i+1}/{config['max_rounds']} ---")
            
            new_segments = []
            for segment in segments:
                if len(segment['content']) <= config['max_segment_size']:
                    new_segments.append(segment)
                    continue
                
                sub_segments = self._split_by_attention(segment)
                new_segments.extend(sub_segments)
            
            if len(new_segments) == len(segments):
                logger.info("Convergence reached.")
                break
            
            segments = new_segments
            
        return segments

    def _split_by_attention(self, segment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Uses attention to find semantic boundaries."""
        content = segment['content']
        logger.info(f"Applying attention-based splitting for segment of size {len(content)}")
        try:
            attention_results = self.attention_extractor.extract_attention_for_tape_splitting([content])
            
            if attention_results and attention_results.get('window_patterns'):
                window_data = attention_results['window_patterns'][0]
                boundaries = window_data.get('optimal_boundaries', [])
                
                if boundaries:
                    logger.info(f"Found {len(boundaries)} optimal boundaries: {boundaries}")
                    split_points = [0] + boundaries + [len(content)]
                    sub_segments = []
                    for i in range(len(split_points) - 1):
                        start, end = split_points[i], split_points[i+1]
                        sub_content = content[start:end]
                        if sub_content.strip():
                            sub_segments.append({'content': sub_content, 'level': segment['level'] + 1})
                    return sub_segments
        except Exception as e:
            logger.error(f"Attention-based splitting failed: {e}. Falling back to simpler method.")

        logger.info(f"Applying fallback splitting for segment of size {len(content)}")
        midpoint = len(content) // 2
        return [
            {'content': content[:midpoint], 'level': segment['level'] + 1},
            {'content': content[midpoint:], 'level': segment['level'] + 1},
        ]

    def _process_single_pass(self, text: str, rules: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Architecturally correct single-pass processing using hierarchical partitioning."""
        logger.info("Starting correct single-pass processing with Hierarchical Partitioner...")
        start_time = datetime.now()

        with Timer("Hierarchical Partitioning"):
            segments = self._hierarchical_partition(text)
        logger.info(f"Created {len(segments)} optimal segments.")

        all_nodes = []
        for i, segment in enumerate(segments):
            all_nodes.append({
                'id': f'node_{i}',
                'content': segment['content'],
                'level': segment['level']
            })
        
        all_edges = []
        for i in range(len(all_nodes) - 1):
            all_edges.append({
                'source': all_nodes[i]['id'],
                'target': all_nodes[i+1]['id'],
                'weight': 1.0,
                'type': 'sequential'
            })

        logger.info(f"Classifying {len(all_nodes)} nodes...")
        if all_nodes:
            graph_for_classification = {"nodes": all_nodes, "edges": all_edges}
            kg_manager = KnowledgeGraphManager(graph_for_classification)
            classified_nodes = kg_manager.classify_nodes()
            
            logger.info(f"Classifying relationships for {len(all_edges)} edges...")
            all_edges = self._classify_edge_relationships(classified_nodes, all_edges)
        else:
            classified_nodes = []

        logger.info("Condensing graph...")
        if all_nodes:
            # Assuming KnowledgeGraphManager is instantiated with the graph
            kg_manager = KnowledgeGraphManager({"nodes": classified_nodes, "edges": all_edges})
            
            # We need an embedding model for condensation
            embedding_model = None
            try:
                from sentence_transformers import SentenceTransformer
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                logger.warning("SentenceTransformers not installed, cannot condense text nodes.")

            if embedding_model:
                with Timer("Graph Condensation"):
                    classified_nodes, all_edges = kg_manager.condense_graph(classified_nodes, all_edges, self.ollama_extractor, embedding_model)
        
        logger.info("Building final hierarchical graph...")
        hierarchical_nodes, tree_edges = self.hierarchical_builder.build_hierarchy(condensed_nodes, condensed_edges)
        
logger.info("Reassembling document from graph...")
# Example of using a specific strategy, e.g., 'layered_assembly'
reassembled = self.graph_reassembler.reassemble(hierarchical_nodes, tree_edges, strategy="layered_assembly", original_document=text)

        processing_time = (datetime.now() - start_time).total_seconds()

        return {
            'mode': 'hierarchical-single-pass',
            'input_length': len(text),
            'nodes': len(classified_nodes),
            'edges': len(all_edges),
            'processing_time': processing_time,
            'output': reassembled,
            'metadata': {
                'model': 'QwQ-32B',
                'architecture': 'Hierarchical-Direct-to-Graph',
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
        
        for edge in edges:
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
                
                for i, node in enumerate(nodes):
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
        '--mode',
        choices=['single-pass'],
        default='single-pass',
        help='Processing mode'
    )
    
    parser.add_argument('--input', '-i', help='Input text file path')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--demo', choices=list(DEMO_CONFIGS.keys()) + ['layered_context_file'],
                       help='Use demo content')
    
    args = parser.parse_args()
    
    config = get_config(mode=args.mode, model_type='ollama')
    
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
        # Pre-load the model once
        logger.info("Pre-loading QwQ model...")
        qwq_path = config['paths']['project_root'] / 'qwq.gguf'
        if not qwq_path.exists():
            raise FileNotFoundError(f"QwQ model not found at {qwq_path}")
            
        from utils.llm_connection_pool import get_global_pool
        connection_pool = get_global_pool()
        ollama_extractor = connection_pool.get_connection(
            str(qwq_path), "ollama", device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Pass the pre-loaded extractor to the processor
        processor = FullMasterProcessor(config, ollama_extractor=ollama_extractor)
        logger.info(f"Starting {args.mode} processing with pre-loaded QwQ...")
        results = processor.process_text(text)
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
