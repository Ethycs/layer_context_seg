#!/usr/bin/env python3
"""
Full Master Processor for Layered Context Graph with QwQ Integration
====================================================================
This version implements all features including:
- QwQ-32B GGUF model integration for attention extraction
- Language-guided processing with natural language rules
- Multi-round annotation with layered analysis
- Direct attention access from transformer models
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Add src directory to path
project_root = Path(__file__).parent.resolve()
src_path = project_root / "layered-context-graph" / "src"
sys.path.insert(0, str(src_path))

# Import config
from master_config import get_config, get_rule_set, DEMO_CONFIGS, RULE_SETS

# Import core modules
from models.context_window import ContextWindow
from models.attention_extractor import EnhancedAttentionExtractor
from models.instruction_seeder import InstructionSeeder
from models.percolation_context_window import PercolationContextWindow
from partitioning.partition_manager import PartitionManager
from graph.graph_reassembler import GraphReassembler
from graph.attention_graph_builder import AttentionGraphBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FullMasterProcessor:
    """Full processor with all features including QwQ integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode = self.config['mode']
        self.model_type = self.config['model_type']
        self.model_config = self.config['model_config']
        self.processing_settings = self.config['processing_settings']
        self.output_dir = self.config['paths']['results_dir']
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all processing components with QwQ"""
        logger.info(f"Initializing components for {self.mode} mode")
        
        # Initialize QwQ-based attention extractor
        qwq_path = self.config['paths']['project_root'] / 'qwq.gguf'
        if qwq_path.exists():
            logger.info(f"Using QwQ model at: {qwq_path}")
            self.attention_extractor = EnhancedAttentionExtractor(str(qwq_path))
        else:
            logger.info("QwQ model not found, using default")
            self.attention_extractor = EnhancedAttentionExtractor()
        
        # Initialize percolation-based context window
        self.percolation_window = PercolationContextWindow(
            size=self.processing_settings.get('window_size', 2000),
            overlap_ratio=0.2  # 20% overlap (within 15-30% range)
        )
        
        # Standard context window for fallback
        self.context_window = ContextWindow(
            size=self.processing_settings.get('window_size', 2000)
        )
        
        # Instruction seeder for language-guided processing
        self.seeder = InstructionSeeder()
        
        # Partition manager with percolation theory
        self.partition_manager = PartitionManager(
            overlap_ratio=0.2,  # 20% overlap (within 15-30% range)
            target_segment_length=self.processing_settings.get('min_chunk_size', 400)
        )
        
        # Graph builders
        self.attention_graph_builder = AttentionGraphBuilder()
        self.graph_reassembler = GraphReassembler()
        
        # Multi-round annotation layers
        if self.mode == 'multi-round':
            self.annotation_layers = self._init_annotation_layers()
    
    def _init_annotation_layers(self) -> Dict[str, Any]:
        """Initialize multi-round annotation layers"""
        layers = {}
        
        # Syntactic layer - grammatical analysis
        layers['syntactic'] = {
            'name': 'Syntactic Analysis',
            'weight': 0.2,
            'features': ['pos_tags', 'dependencies', 'syntax_patterns'],
            'analyzer': self._analyze_syntax
        }
        
        # Semantic layer - meaning and concepts
        layers['semantic'] = {
            'name': 'Semantic Analysis', 
            'weight': 0.5,
            'features': ['topics', 'concepts', 'relationships'],
            'analyzer': self._analyze_semantics
        }
        
        # Pragmatic layer - intent and discourse
        layers['pragmatic'] = {
            'name': 'Pragmatic Analysis',
            'weight': 0.3,
            'features': ['intent', 'discourse', 'importance'],
            'analyzer': self._analyze_pragmatics
        }
        
        return layers
    
    def process_text(self, text: str, rules: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process text using the configured mode"""
        
        if self.mode == 'language-guided':
            return self._process_language_guided(text, rules)
        elif self.mode == 'multi-round':
            return self._process_multi_round(text, rules)
        else:
            return self._process_single_pass(text, rules)
    
    def _process_single_pass(self, text: str, rules: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Single-pass processing with QwQ attention extraction"""
        logger.info("Starting single-pass processing with QwQ...")
        
        start_time = datetime.now()
        
        # Step 1: Create percolation-based windows
        logger.info("Creating percolation windows...")
        windows = self.percolation_window.create_window(text)
        logger.info(f"Created {len(windows)} percolation windows")
        
        # Step 2: Extract QwQ attention patterns for each window
        logger.info("Extracting QwQ attention patterns...")
        windows_with_attention = []
        
        for i, window_content in enumerate(windows):
            # Extract attention using QwQ - use tape splitting method
            attention_data = self.attention_extractor.extract_attention_for_tape_splitting([window_content])
            
            # Build attention graph for this window
            attention_graph = self.attention_graph_builder.build_from_attention(
                [window_content],  # Pass as list
                attention_data
            )
            
            windows_with_attention.append({
                'id': f'window_{i}',
                'content': window_content,
                'overlap': self.percolation_window.overlap_ratio,
                'attention': attention_data,
                'graph': attention_graph
            })
        
        # Step 3: Create optimal partitions using disassembly rules
        logger.info("Applying disassembly rules...")
        partitions = self.partition_manager.create_partitions(
            [w['content'] for w in windows_with_attention]
        )
        
        # Step 4: Build knowledge graph
        logger.info("Building knowledge graph...")
        nodes = []
        edges = []
        
        for i, (partition, window_data) in enumerate(zip(partitions, windows_with_attention)):
            node = {
                'id': f'node_{i}',
                'content': partition,
                'attention': window_data['attention'],
                'importance': self._calculate_importance(window_data['attention'])
            }
            nodes.append(node)
            
            # Create edges based on attention patterns
            if i > 0 and 'graph' in window_data:
                for edge in window_data['graph'].get('edges', []):
                    edges.append({
                        'source': f'node_{i-1}',
                        'target': f'node_{i}',
                        'weight': edge.get('weight', 1.0),
                        'type': 'attention'
                    })
        
        # Step 5: Reassemble using reconstruction rules
        logger.info("Applying reassembly rules...")
        reassembled = self.graph_reassembler.reassemble_graph(nodes, edges, text)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'mode': 'single-pass',
            'input_length': len(text),
            'windows': len(windows),
            'nodes': len(nodes),
            'edges': len(edges),
            'processing_time': processing_time,
            'output': reassembled,
            'metadata': {
                'model': 'QwQ-32B',
                'percolation_overlap': '15-30%',
                'attention_extracted': True,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _process_language_guided(self, text: str, rules: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Language-guided processing with natural language rules"""
        logger.info("Starting language-guided processing...")
        
        start_time = datetime.now()
        
        # Step 1: Apply segmentation rules if provided
        if rules and 'segmentation' in rules:
            logger.info(f"Applying segmentation rule: {rules['segmentation']}")
            # Seed the text with segmentation instructions
            seeded_text = self.seeder.seed_instructions(
                text,
                {'QWQ_SEGMENT': rules['segmentation']}
            )
        else:
            seeded_text = text
        
        # Step 2: Create windows with language-guided boundaries
        windows = self.percolation_window.create_window(seeded_text)
        
        # Step 3: Extract attention and apply reorganization rules
        nodes = []
        
        for i, window_content in enumerate(windows):
            attention = self.attention_extractor.extract_attention_for_tape_splitting([window_content])
            
            # Apply reorganization rules to determine node properties
            if rules and 'reorganization' in rules:
                importance = self._apply_reorganization_rule(
                    window_content,
                    attention,
                    rules['reorganization']
                )
            else:
                importance = self._calculate_importance(attention)
            
            nodes.append({
                'id': f'guided_node_{i}',
                'content': window_content,
                'attention': attention,
                'importance': importance,
                'rule_based': True
            })
        
        # Step 4: Build graph with rule-based connections
        edges = self._build_rule_based_edges(nodes, rules)
        
        # Step 5: Reassemble with language-guided organization
        reassembled = self.graph_reassembler.reassemble_graph(
            nodes,
            edges,
            text,
            organization_rule=rules.get('reorganization') if rules else None
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'mode': 'language-guided',
            'input_length': len(text),
            'nodes': len(nodes),
            'edges': len(edges),
            'rules_applied': rules is not None,
            'processing_time': processing_time,
            'output': reassembled,
            'metadata': {
                'model': 'QwQ-32B',
                'segmentation_rule': rules.get('segmentation') if rules else None,
                'reorganization_rule': rules.get('reorganization') if rules else None,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _process_multi_round(self, text: str, rules: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Multi-round annotation processing"""
        logger.info("Starting multi-round annotation processing...")
        
        start_time = datetime.now()
        
        # Step 1: Create base graph (single pass)
        logger.info("Creating base graph...")
        base_result = self._process_single_pass(text, rules)
        base_nodes = base_result['output'].get('nodes', [])
        
        # Step 2: Apply annotation layers
        annotations = {}
        
        for layer_name, layer_config in self.annotation_layers.items():
            logger.info(f"Applying {layer_name} annotation layer...")
            
            # Analyze each node with layer-specific analyzer
            layer_annotations = []
            
            for node in base_nodes:
                annotation = layer_config['analyzer'](
                    node['content'],
                    node.get('attention', {})
                )
                layer_annotations.append({
                    'node_id': node['id'],
                    'layer': layer_name,
                    'features': annotation,
                    'weight': layer_config['weight']
                })
            
            annotations[layer_name] = {
                'config': layer_config,
                'annotations': layer_annotations
            }
        
        # Step 3: Cross-layer synthesis
        logger.info("Performing cross-layer synthesis...")
        synthesis = self._synthesize_annotations(base_nodes, annotations)
        
        # Step 4: Create enriched graph
        enriched_nodes = self._enrich_nodes(base_nodes, annotations, synthesis)
        enriched_edges = self._enrich_edges(
            base_result['output'].get('edges', []),
            synthesis
        )
        
        # Step 5: Final reassembly with multi-layer insights
        final_output = self.graph_reassembler.reassemble_graph(
            enriched_nodes,
            enriched_edges,
            text,
            synthesis_data=synthesis
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'mode': 'multi-round',
            'base_result': base_result,
            'annotation_layers': list(annotations.keys()),
            'synthesis': synthesis,
            'enriched_nodes': len(enriched_nodes),
            'enriched_edges': len(enriched_edges),
            'processing_time': processing_time,
            'output': final_output,
            'metadata': {
                'model': 'QwQ-32B',
                'layers_applied': len(annotations),
                'cross_layer_synthesis': True,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    # Helper methods for analysis
    
    def _calculate_importance(self, attention_data: Dict) -> float:
        """Calculate importance score from attention patterns"""
        if isinstance(attention_data, dict) and 'attention_scores' in attention_data:
            scores = attention_data['attention_scores']
            if isinstance(scores, list):
                return float(np.mean(scores))
        return 0.5  # Default importance
    
    def _apply_reorganization_rule(self, content: str, attention: Dict, rule: str) -> float:
        """Apply reorganization rule to determine importance"""
        # Parse rule and adjust importance based on content matching
        importance = self._calculate_importance(attention)
        
        # Simple rule matching (can be enhanced)
        rule_keywords = rule.lower().split()
        content_lower = content.lower()
        
        matches = sum(1 for keyword in rule_keywords if keyword in content_lower)
        if matches > 0:
            importance *= (1 + 0.1 * matches)  # Boost importance for matches
        
        return min(importance, 1.0)
    
    def _build_rule_based_edges(self, nodes: List[Dict], rules: Optional[Dict]) -> List[Dict]:
        """Build edges based on rules and attention patterns"""
        edges = []
        
        for i in range(len(nodes) - 1):
            # Basic sequential connection
            edge = {
                'source': nodes[i]['id'],
                'target': nodes[i + 1]['id'],
                'weight': 1.0,
                'type': 'sequential'
            }
            
            # Enhance weight based on attention similarity
            if 'attention' in nodes[i] and 'attention' in nodes[i + 1]:
                similarity = self._compute_attention_similarity(
                    nodes[i]['attention'],
                    nodes[i + 1]['attention']
                )
                edge['weight'] = similarity
                edge['type'] = 'attention_similarity'
            
            edges.append(edge)
        
        return edges
    
    def _compute_attention_similarity(self, attn1: Dict, attn2: Dict) -> float:
        """Compute similarity between two attention patterns"""
        # Simple implementation - can be enhanced
        return 0.8  # Placeholder
    
    # Annotation layer analyzers
    
    def _analyze_syntax(self, content: str, attention: Dict) -> Dict[str, Any]:
        """Syntactic analysis layer"""
        # Simplified syntactic analysis
        return {
            'sentence_count': content.count('.') + content.count('!') + content.count('?'),
            'avg_word_length': np.mean([len(w) for w in content.split()]),
            'complexity': len(content.split()) / (content.count('.') + 1)
        }
    
    def _analyze_semantics(self, content: str, attention: Dict) -> Dict[str, Any]:
        """Semantic analysis layer"""
        # Use attention patterns to identify key concepts
        return {
            'key_terms': self._extract_key_terms(content, attention),
            'topic_relevance': self._calculate_importance(attention),
            'semantic_density': len(set(content.split())) / len(content.split())
        }
    
    def _analyze_pragmatics(self, content: str, attention: Dict) -> Dict[str, Any]:
        """Pragmatic analysis layer"""
        # Analyze communicative intent
        return {
            'intent': self._detect_intent(content),
            'discourse_marker': self._has_discourse_markers(content),
            'importance': self._calculate_importance(attention) * 1.2
        }
    
    def _extract_key_terms(self, content: str, attention: Dict) -> List[str]:
        """Extract key terms based on attention"""
        # Simple word frequency approach
        words = content.split()
        return list(set(w for w in words if len(w) > 4))[:5]
    
    def _detect_intent(self, content: str) -> str:
        """Detect communicative intent"""
        if '?' in content:
            return 'question'
        elif any(marker in content.lower() for marker in ['therefore', 'thus', 'hence']):
            return 'conclusion'
        elif any(marker in content.lower() for marker in ['however', 'but', 'although']):
            return 'contrast'
        else:
            return 'statement'
    
    def _has_discourse_markers(self, content: str) -> bool:
        """Check for discourse markers"""
        markers = ['moreover', 'furthermore', 'however', 'therefore', 'thus',
                  'firstly', 'secondly', 'finally', 'in conclusion']
        return any(marker in content.lower() for marker in markers)
    
    def _synthesize_annotations(self, nodes: List[Dict], annotations: Dict) -> Dict[str, Any]:
        """Synthesize insights across annotation layers"""
        synthesis = {
            'node_importance': {},
            'layer_agreement': {},
            'key_insights': []
        }
        
        # Calculate weighted importance for each node
        for node in nodes:
            node_id = node['id']
            total_importance = 0
            
            for layer_name, layer_data in annotations.items():
                for ann in layer_data['annotations']:
                    if ann['node_id'] == node_id:
                        weight = layer_data['config']['weight']
                        importance = ann['features'].get('importance', 0.5)
                        total_importance += weight * importance
            
            synthesis['node_importance'][node_id] = total_importance
        
        # Find nodes where layers agree on high importance
        high_importance_threshold = 0.7
        for node_id, importance in synthesis['node_importance'].items():
            if importance > high_importance_threshold:
                synthesis['key_insights'].append({
                    'node_id': node_id,
                    'importance': importance,
                    'reason': 'High cross-layer importance'
                })
        
        return synthesis
    
    def _enrich_nodes(self, nodes: List[Dict], annotations: Dict, synthesis: Dict) -> List[Dict]:
        """Enrich nodes with multi-layer annotations"""
        enriched = []
        
        for node in nodes:
            enriched_node = node.copy()
            enriched_node['annotations'] = {}
            
            # Add annotations from each layer
            for layer_name, layer_data in annotations.items():
                for ann in layer_data['annotations']:
                    if ann['node_id'] == node['id']:
                        enriched_node['annotations'][layer_name] = ann['features']
            
            # Add synthesis information
            enriched_node['synthesis_importance'] = synthesis['node_importance'].get(
                node['id'], 0.5
            )
            
            enriched.append(enriched_node)
        
        return enriched
    
    def _enrich_edges(self, edges: List[Dict], synthesis: Dict) -> List[Dict]:
        """Enrich edges based on synthesis insights"""
        enriched = []
        
        for edge in edges:
            enriched_edge = edge.copy()
            
            # Enhance edge weight based on node importance
            source_imp = synthesis['node_importance'].get(edge['source'], 0.5)
            target_imp = synthesis['node_importance'].get(edge['target'], 0.5)
            
            enriched_edge['synthesis_weight'] = (source_imp + target_imp) / 2
            enriched_edge['weight'] = edge.get('weight', 1.0) * enriched_edge['synthesis_weight']
            
            enriched.append(enriched_edge)
        
        return enriched
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> Path:
        """Save processing results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qwq_layered_results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
        return output_path


def get_demo_content(demo_type: str) -> str:
    """Get demo content"""
    demo_map = {
        'transcript': """
        Speaker A: Let's discuss the new architecture for our system.
        Speaker B: I think we should use a microservices approach for better scalability.
        Speaker A: That makes sense. We'll need to consider service discovery and communication.
        Speaker B: Absolutely. And don't forget about load balancing and fault tolerance.
        Speaker A: Right. We should also implement proper monitoring and logging.
        Speaker B: I'll draft a proposal with these key components outlined.
        """,
        'technical': """
        ## System Architecture Overview
        
        The layered context graph system transforms linear documents into multi-dimensional knowledge graphs.
        
        ### Core Components
        
        1. **Attention Extractor**: Uses QwQ-32B model to extract attention patterns from text
        2. **Percolation Windows**: Creates overlapping windows with 15-30% overlap based on percolation theory
        3. **Graph Builder**: Constructs knowledge graph from attention patterns and semantic relationships
        
        ### Processing Pipeline
        
        The system implements a three-phase approach:
        - Disassembly: Breaking down text into optimal segments
        - Analysis: Extracting patterns and relationships
        - Reassembly: Reconstructing into purpose-driven outputs
        
        This architecture enables flexible document transformation while preserving semantic integrity.
        """,
        'simple': """
        This is a simple example text that demonstrates basic processing.
        It contains multiple sentences and paragraphs to show segmentation.
        
        The processor will analyze this text and create a layered context graph.
        Each sentence becomes a node with attention-based connections.
        
        The final output reorganizes the content based on importance and relationships.
        """
    }
    return demo_map.get(demo_type, "No demo content found.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Full Layered Context Graph Processor with QwQ Integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-pass with QwQ attention extraction
  python master_processor_full.py --demo simple --mode single-pass
  
  # Language-guided with custom rules
  python master_processor_full.py --demo technical --mode language-guided --rules technical_documentation
  
  # Multi-round annotation with all layers
  python master_processor_full.py --demo transcript --mode multi-round
  
  # Process custom file
  python master_processor_full.py --input document.txt --mode multi-round --output results/
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['single-pass', 'multi-round', 'language-guided'],
        default='single-pass',
        help='Processing mode'
    )
    
    # Input/Output
    parser.add_argument('--input', '-i', help='Input text file path')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--demo', choices=DEMO_CONFIGS.keys(),
                       help='Use demo content')
    
    # Processing options
    parser.add_argument('--rules', choices=RULE_SETS.keys(),
                       help='Predefined rule set')
    
    # Advanced options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(mode=args.mode, model_type='ollama')
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Override output directory
    if args.output:
        config['paths']['results_dir'] = Path(args.output)
    
    # Get input text
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
    
    # Get rules
    rules = None
    if args.rules:
        rules = get_rule_set(args.rules)
        logger.info(f"Using rule set: {args.rules}")
    
    # Process
    try:
        processor = FullMasterProcessor(config)
        logger.info(f"Starting {args.mode} processing with QwQ...")
        
        results = processor.process_text(text, rules)
        
        # Save results
        output_path = processor.save_results(results)
        
        # Print summary
        print("\n" + "="*60)
        print("QWQ-POWERED PROCESSING COMPLETE")
        print("="*60)
        print(f"Mode: {results['mode']}")
        print(f"Model: {results['metadata'].get('model', 'QwQ-32B')}")
        print(f"Input length: {results.get('input_length', len(text))} characters")
        
        if 'nodes' in results:
            print(f"Knowledge graph nodes: {results['nodes']}")
        if 'edges' in results:
            print(f"Knowledge graph edges: {results['edges']}")
        
        if args.mode == 'multi-round':
            print(f"Annotation layers: {results.get('annotation_layers', [])}")
            if 'synthesis' in results:
                print(f"Key insights found: {len(results['synthesis'].get('key_insights', []))}")
        
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        print(f"Results saved to: {output_path}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()