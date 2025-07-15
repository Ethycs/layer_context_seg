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
from models.percolation_context_window import PercolationContextWindow
from partitioning.partition_manager import PartitionManager
from graph.graph_reassembler import GraphReassembler
from graph.attention_graph_builder import AttentionGraphBuilder
from graph.hierarchical_graph_builder import HierarchicalGraphBuilder

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
    
    def __init__(self, config: Dict[str, Any], graph_config: GraphConfig = None):
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
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all processing components with QwQ"""
        logger.info(f"Initializing components for {self.mode} mode")
        
        # Initialize QwQ-based attention extractor
        qwq_path = self.config['paths']['project_root'] / 'qwq.gguf'
        if qwq_path.exists():
            logger.info(f"Using QwQ model at: {qwq_path}")
            # Use OllamaModelExtractor for GGUF files with GPU support
            from models.ollama_extractor import OllamaModelExtractor
            self.ollama_extractor = OllamaModelExtractor(str(qwq_path), device=self.device)
            self.attention_extractor = EnhancedAttentionExtractor(str(qwq_path))
        else:
            logger.info("QwQ model not found, using default")
            self.ollama_extractor = None
            self.attention_extractor = EnhancedAttentionExtractor()
        
        # Initialize percolation-based context window
        # Use larger window size for documents (not conversations)
        default_window_size = 8192 if self.mode != 'conversation' else 2000
        self.percolation_window = PercolationContextWindow(
            size=self.processing_settings.get('window_size', default_window_size),
            overlap_ratio=0.2  # 20% overlap (within 15-30% range)
        )
        
        # Standard context window for fallback
        self.context_window = ContextWindow(
            size=self.processing_settings.get('window_size', 2000)
        )
        
        # Instruction seeder for language-guided processing
        self.seeder = InstructionSeeder()
        
        # Partition manager with percolation theory
        if self.graph_config:
            # Try formatting-preserving manager first
            try:
                from partitioning.formatting_preserving_partition_manager import FormattingPreservingPartitionManager
                self.partition_manager = FormattingPreservingPartitionManager(
                    config=self.graph_config.disassembly
                )
                logger.info("Using formatting-preserving partition manager with centralized config")
            except ImportError:
                # Try content-preserving manager
                try:
                    from partitioning.content_preserving_partition_manager import ContentPreservingPartitionManager
                    self.partition_manager = ContentPreservingPartitionManager(
                        min_segment_length=self.graph_config.disassembly.min_segment_length,
                        target_segment_length=self.graph_config.disassembly.target_segment_length,
                        max_segment_length=self.graph_config.disassembly.max_segment_length,
                        overlap_ratio=self.graph_config.disassembly.overlap_ratio
                    )
                    logger.info("Using content-preserving partition manager")
                except ImportError:
                    # Fall back to standard manager
                    self.partition_manager = PartitionManager(
                        overlap_ratio=self.graph_config.disassembly.overlap_ratio,
                        target_segment_length=self.graph_config.disassembly.target_segment_length
                    )
                    logger.info("Using standard partition manager")
        else:
            # No graph config - use defaults
            try:
                from partitioning.content_preserving_partition_manager import ContentPreservingPartitionManager
                self.partition_manager = ContentPreservingPartitionManager(
                    min_segment_length=800,
                    target_segment_length=1500,
                    max_segment_length=3000,
                    overlap_ratio=0.2
                )
                logger.info("Using content-preserving partition manager")
            except ImportError:
                self.partition_manager = PartitionManager(
                    overlap_ratio=0.2,
                    target_segment_length=self.processing_settings.get('min_chunk_size', 400)
                )
                logger.info("Using standard partition manager")
        
        # Graph builders - prefer GPU-enabled versions when available
        try:
            from graph.torch_attention_graph_builder import TorchAttentionGraphBuilder
            self.torch_graph_builder = TorchAttentionGraphBuilder(device=self.device)
            logger.info("Using GPU-accelerated TorchAttentionGraphBuilder")
        except ImportError:
            self.torch_graph_builder = None
            logger.info("TorchAttentionGraphBuilder not available")
        
        # Pass attention extractor to graph builder for attention-based edge detection
        self.attention_graph_builder = AttentionGraphBuilder(
            attention_extractor=self.attention_extractor
        )
        
        # Graph reassembler with formatting preservation
        if self.graph_config:
            try:
                from graph.formatting_preserving_reassembler import FormattingPreservingReassembler
                self.graph_reassembler = FormattingPreservingReassembler(
                    config=self.graph_config.reconstruction
                )
                logger.info("Using formatting-preserving reassembler with centralized config")
            except ImportError:
                try:
                    from graph.enhanced_graph_reassembler import EnhancedGraphReassembler
                    self.graph_reassembler = EnhancedGraphReassembler()
                    logger.info("Using enhanced graph reassembler")
                except ImportError:
                    self.graph_reassembler = GraphReassembler()
                    logger.info("Using standard graph reassembler")
        else:
            # No graph config - use best available
            try:
                from graph.enhanced_graph_reassembler import EnhancedGraphReassembler
                self.graph_reassembler = EnhancedGraphReassembler()
                logger.info("Using enhanced graph reassembler for rich reconstruction")
            except ImportError:
                self.graph_reassembler = GraphReassembler()
                logger.info("Using standard graph reassembler")
            
        # Transformer-based document builder for progressive generation
        try:
            from synthesis.transformer_document_builder import TransformerDocumentBuilder
            self.document_builder = TransformerDocumentBuilder()
            logger.info("Transformer document builder available for progressive generation")
        except ImportError:
            self.document_builder = None
            logger.info("Transformer document builder not available")
            
        self.hierarchical_builder = HierarchicalGraphBuilder()
        
        # Initialize TorchSpectralProcessor if available and mode requires it
        self.spectral_processor = None
        if SPECTRAL_AVAILABLE and (self.mode == 'spectral-hybrid' or self.config.get('use_spectral', False)):
            try:
                qwq_path = str(self.config['paths']['project_root'] / 'qwq.gguf') if qwq_path.exists() else None
                self.spectral_processor = TorchSpectralProcessor(
                    qwq_model_path=qwq_path,
                    device=str(self.device)
                )
                logger.info("Initialized TorchSpectralProcessor for hybrid processing")
            except Exception as e:
                logger.warning(f"Failed to initialize TorchSpectralProcessor: {e}")
                self.spectral_processor = None
        
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
        elif self.mode == 'spectral-hybrid':
            return self._process_spectral_hybrid(text, rules)
        else:
            return self._process_single_pass(text, rules)
    
    def _process_single_pass(self, text: str, rules: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Enhanced single-pass processing with language-guided seeding and multi-round analysis"""
        logger.info("Starting enhanced single-pass processing with QwQ...")
        
        start_time = datetime.now()
        
        # Step 1: Apply language-guided instruction seeding
        logger.info("Applying language-guided instruction seeding...")
        if rules and 'segmentation' in rules:
            logger.info(f"Using segmentation rule: {rules['segmentation']}")
            if rules['segmentation'] == 'conversation_boundaries':
                seeded_text = self.seeder.seed_instructions(text, density=0.15)
            else:
                seeded_text = self.seeder.seed_instructions(text, density=0.1)
        else:
            # Apply default intelligent seeding
            seeded_text = self.seeder.seed_instructions(text, density=0.1)
        
        # Step 2: Create percolation-based windows from seeded text
        logger.info("Creating percolation windows from seeded text...")
        windows = self.percolation_window.create_window(seeded_text)
        logger.info(f"Created {len(windows)} percolation windows")
        
        # Store original text for formatting extraction
        original_text = text
        
        # Step 2: Extract QwQ attention patterns for each window
        logger.info("Extracting QwQ attention patterns...")
        windows_with_attention = []
        
        for i, window_content in enumerate(windows):
            # Extract attention using QwQ - prefer GPU-enabled extractor
            if self.ollama_extractor:
                # Use GPU-enabled OllamaModelExtractor
                attention_patterns = self.ollama_extractor.get_attention_patterns(window_content)
                # Convert to expected format
                attention_data = {
                    'attention_scores': [1.0] * len(window_content.split()),
                    'attention_patterns': attention_patterns
                }
            else:
                # Fallback to standard extraction
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
        
        # Step 3: Prepare data for direct graph construction
        logger.info("Preparing data for direct graph construction...")
        all_contents = [w['content'] for w in windows_with_attention]
        
        # Step 4: Build knowledge graph directly from attention patterns with formatting preservation
        if self.torch_graph_builder and len(windows_with_attention) > 0:
            logger.info("Building graph directly from attention patterns using TorchAttentionGraphBuilder...")
            
            # Extract formatting information using FormattingPreservingPartitionManager
            logger.info("Extracting formatting information for each window...")
            windows_with_formatting = []
            for i, window_data in enumerate(windows_with_attention):
                content = window_data['content']
                
                # Extract formatting directly from the content (which should now preserve formatting)
                formatting = self._extract_basic_formatting(content)
                
                windows_with_formatting.append({
                    **window_data,
                    'formatting': formatting
                })
            
            # Create unified attention tensor from all windows
            unified_attention_tensor = self._create_unified_attention(windows_with_formatting)
            
            # Build graph directly using the TorchAttentionGraphBuilder
            graph_output = self.torch_graph_builder.forward(
                attention_tensor=unified_attention_tensor,
                text_segments=all_contents
            )
            nodes = graph_output.get('nodes', [])
            edges = graph_output.get('edges', [])
            
            # Enhance nodes with additional metadata INCLUDING formatting
            for i, node in enumerate(nodes):
                window_data = windows_with_formatting[i] if i < len(windows_with_formatting) else windows_with_formatting[-1]
                
                # Apply reorganization rules if provided
                if rules and 'reorganization' in rules:
                    importance = self._apply_reorganization_rule(
                        node['content'],
                        window_data['attention'],
                        rules['reorganization']
                    )
                else:
                    importance = self._calculate_importance(window_data['attention'])
                
                node.update({
                    'id': f'node_{i}',
                    'attention': window_data['attention'],
                    'importance': importance,
                    'segment_type': self._classify_segment_type(node['content']),
                    'formatting': window_data.get('formatting', {}),  # PRESERVE FORMATTING
                    'reconstruction_layer': 0,
                    'layer_name': 'Layer 0',
                    'rule_based': rules is not None
                })
                
        else:
            # Fallback: create nodes directly from windows if TorchAttentionGraphBuilder not available
            logger.info("TorchAttentionGraphBuilder not available, using fallback node creation...")
            nodes = []
            edges = []
            
            for i, window_data in enumerate(windows_with_attention):
                content = window_data['content']
                node = {
                    'id': f'node_{i}',
                    'content': content,
                    'attention': window_data['attention'],
                    'importance': self._calculate_importance(window_data['attention']),
                    'segment_type': self._classify_segment_type(content),
                    'reconstruction_layer': 0,
                    'layer_name': 'Layer 0'
                }
                nodes.append(node)
                
                # Create sequential edges between nodes
                if i > 0:
                    edges.append({
                        'source': f'node_{i-1}',
                        'target': f'node_{i}',
                        'weight': 1.0,
                        'type': 'sequential'
                    })
        
        # Step 5: Apply multi-round annotation layers if available
        if hasattr(self, 'annotation_layers') and self.annotation_layers:
            logger.info("Applying multi-round annotation layers...")
            annotations = {}
            
            for layer_name, layer_config in self.annotation_layers.items():
                logger.info(f"Applying {layer_name} annotation layer...")
                
                # Analyze each node with layer-specific analyzer
                layer_annotations = []
                
                for node in nodes:
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
            
            # Cross-layer synthesis
            logger.info("Performing cross-layer synthesis...")
            synthesis = self._synthesize_annotations(nodes, annotations)
            
            # Enrich nodes and edges with multi-layer insights
            nodes = self._enrich_nodes(nodes, annotations, synthesis)
            edges = self._enrich_edges(edges, synthesis)
        
        # Step 6: Enhanced edge analysis using available components
        logger.info("Applying enhanced edge analysis...")
        
        # Import edge analysis components
        try:
            from graph.conversation_edge_types import ConversationEdgeAnalyzer
            from graph.enhanced_edge_detector import EnhancedEdgeDetector
            from graph.attention_based_edge_detector import AttentionBasedEdgeDetector
            
            # Apply conversation-specific edge analysis
            conversation_analyzer = ConversationEdgeAnalyzer(self.attention_extractor)
            conversation_edges = conversation_analyzer.create_conversation_edges(nodes, use_attention=True)
            
            # Apply enhanced edge detection
            enhanced_detector = EnhancedEdgeDetector()
            enhanced_edges = enhanced_detector.detect_edges(nodes)
            
            # Apply attention-based edge detection
            attention_detector = AttentionBasedEdgeDetector(self.attention_extractor)
            final_edges = attention_detector.enhance_edges_with_attention(enhanced_edges, nodes)
            
            # Use the enriched edges
            edges = final_edges
            logger.info(f"Enhanced edge analysis complete - {len(edges)} edges with rich metadata")
            
        except ImportError as e:
            logger.warning(f"Enhanced edge analysis not available: {e}")
            # Fall back to basic edges
        
        # Step 7: Build hierarchical structure
        logger.info("Building hierarchical graph structure...")
        hierarchical_nodes, tree_edges = self.hierarchical_builder.build_hierarchy(nodes, edges)
        
        # Step 7: Generate analysis report (existing functionality)
        logger.info("Applying reassembly rules...")
        reassembled = self.graph_reassembler.reassemble_graph(hierarchical_nodes, tree_edges, text)
        
        # Step 7: Add synthesis capabilities (actual Tape2 generation)
        logger.info("Synthesis capabilities available - use synthesize_content() method")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'mode': 'enhanced-single-pass',
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
                'language_guided_seeding': True,
                'multi_round_annotations': hasattr(self, 'annotation_layers') and bool(self.annotation_layers),
                'direct_to_graph': True,
                'formatting_preserved': True,
                'rules_applied': rules is not None,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _classify_segment_type(self, content: str) -> str:
        """Classify segment for reconstruction purposes"""
        content_lower = content.lower()
        
        # Look for instruction markers and technical indicators
        if any(marker in content for marker in ['<MATH>', '<QWQ_REASONING>', 'algorithm', 'implementation']):
            return 'technical_core'
        elif any(marker in content for marker in ['<DIALOGUE>', '<QWQ_EXAMPLE>', 'example', 'demonstration']):
            return 'illustrative'
        elif any(word in content_lower for word in ['definition', 'concept', 'introduction', 'overview']):
            return 'foundational'
        elif any(word in content_lower for word in ['application', 'use case', 'practical']):
            return 'application'
        elif any(word in content_lower for word in ['problem', 'issue', 'challenge', 'solution']):
            return 'problem_solving'
        else:
            return 'supporting'
    
    def _process_language_guided(self, text: str, rules: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Language-guided processing with natural language rules"""
        logger.info("Starting language-guided processing...")
        
        start_time = datetime.now()
        
        # Step 1: Apply segmentation rules if provided
        if rules and 'segmentation' in rules:
            logger.info(f"Applying segmentation rule: {rules['segmentation']}")
            # Seed the text with segmentation instructions
            if rules['segmentation'] == 'conversation_boundaries':
                seeded_text = self.seeder.seed_instructions(text, density=0.15)
            else:
                seeded_text = self.seeder.seed_instructions(text, density=0.1)
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
                'rule_based': True,
                'segment_type': self._classify_segment_type(window_content),
                'reconstruction_layer': 0,
                'layer_name': 'Layer 0'
            })
        
        # Step 4: Build graph with rule-based connections
        edges = self._build_rule_based_edges(nodes, rules)
        
        # Step 5: Reassemble with language-guided organization
        reassembled = self.graph_reassembler.reassemble_graph(
            nodes,
            edges,
            text
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
            text
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
    
    def _create_unified_attention(self, windows_with_attention: List[Dict]) -> torch.Tensor:
        """
        Create a unified attention tensor from multiple windows
        
        Args:
            windows_with_attention: List of window data with attention patterns
            
        Returns:
            Unified attention tensor for TorchAttentionGraphBuilder
        """
        if not windows_with_attention:
            return torch.eye(1, device=self.device)  # Identity matrix as fallback
            
        # Extract attention patterns from each window
        attention_matrices = []
        max_seq_len = 0
        
        for window_data in windows_with_attention:
            attention = window_data.get('attention', {})
            
            # Handle different attention formats
            if 'attention_scores' in attention:
                scores = attention['attention_scores']
                seq_len = len(scores)
                # Create self-attention matrix from scores
                attn_matrix = torch.ones(seq_len, seq_len, device=self.device) * 0.1
                for i, score in enumerate(scores):
                    attn_matrix[i, i] = float(score)
                attention_matrices.append(attn_matrix)
                max_seq_len = max(max_seq_len, seq_len)
            elif 'attention_patterns' in attention:
                # Use existing attention patterns
                patterns = attention['attention_patterns']
                if isinstance(patterns, torch.Tensor):
                    attention_matrices.append(patterns)
                    max_seq_len = max(max_seq_len, patterns.size(-1))
                else:
                    # Convert to tensor if needed
                    seq_len = len(str(window_data['content']).split())
                    attn_matrix = torch.eye(seq_len, device=self.device)
                    attention_matrices.append(attn_matrix)
                    max_seq_len = max(max_seq_len, seq_len)
            else:
                # Fallback: create identity matrix based on content length
                seq_len = len(str(window_data['content']).split())
                attn_matrix = torch.eye(seq_len, device=self.device)
                attention_matrices.append(attn_matrix)
                max_seq_len = max(max_seq_len, seq_len)
        
        # Pad matrices to same size and concatenate
        if max_seq_len == 0:
            return torch.eye(1, device=self.device)
            
        padded_matrices = []
        for matrix in attention_matrices:
            current_size = matrix.size(0)
            if current_size < max_seq_len:
                # Pad with zeros
                pad_size = max_seq_len - current_size
                padded = F.pad(matrix, (0, pad_size, 0, pad_size), value=0.0)
                padded_matrices.append(padded)
            else:
                padded_matrices.append(matrix[:max_seq_len, :max_seq_len])
        
        # Stack matrices to create unified attention tensor
        if len(padded_matrices) == 1:
            unified = padded_matrices[0].unsqueeze(0)  # Add batch dimension
        else:
            # Combine multiple attention matrices (average them)
            unified = torch.stack(padded_matrices, dim=0).mean(dim=0).unsqueeze(0)
            
        return unified
    
    def _extract_basic_formatting(self, text: str) -> Dict[str, Any]:
        """
        Extract basic formatting information from text as fallback
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with formatting metadata
        """
        formatting = {
            'indentation': '',
            'line_breaks': text.count('\n'),
            'ends_with_newline': text.endswith('\n'),
            'leading_space': len(text) - len(text.lstrip()),
            'trailing_space': len(text) - len(text.rstrip()),
            'has_bold': '**' in text or '__' in text,
            'has_italic': '*' in text and '**' not in text,
            'has_code': '```' in text or '`' in text,
            'heading_level': None
        }
        
        # Check for markdown headings
        lines = text.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                formatting['heading_level'] = len(stripped) - len(stripped.lstrip('#'))
                break
        
        # Check for list items
        formatting['has_list'] = any(
            line.strip().startswith(('-', '*', '+')) or 
            re.match(r'^\s*\d+\.', line.strip())
            for line in lines
        )
        
        # Check for conversation format
        if any(line.strip().startswith(('User:', 'Claude:', 'Assistant:')) for line in lines):
            formatting['speakers'] = []
            formatting['turn_count'] = 0
            for line in lines:
                if ':' in line:
                    speaker = line.split(':')[0].strip()
                    if speaker in ['User', 'Claude', 'Assistant'] and speaker not in formatting['speakers']:
                        formatting['speakers'].append(speaker)
                        formatting['turn_count'] += 1
        
        # Check for code language
        if '```' in text:
            match = re.search(r'```(\w+)', text)
            if match:
                formatting['language'] = match.group(1)
        
        return formatting
    
    def _find_original_text_for_window(self, original_text: str, processed_content: str, window_index: int) -> str:
        """
        Find the original text segment that corresponds to a processed window
        
        Args:
            original_text: The original unprocessed text
            processed_content: The processed content from the window
            window_index: Index of the window
            
        Returns:
            The original text segment corresponding to the window
        """
        # Try to find the processed content in the original text
        # This is a fuzzy matching approach since the processed content may have been modified
        
        # Clean up the processed content to find a match
        processed_clean = processed_content.strip()
        
        # If the processed content is too short or empty, return it as-is
        if len(processed_clean) < 50:
            return processed_content
        
        # Try to find a substring match in the original text
        # Use the first few words and last few words to find the boundaries
        words = processed_clean.split()
        if len(words) < 5:
            return processed_content
            
        # Create search patterns from first and last words
        first_words = ' '.join(words[:3])
        last_words = ' '.join(words[-3:])
        
        # Find the start position
        start_pos = original_text.find(first_words)
        if start_pos == -1:
            # Try with single word
            start_pos = original_text.find(words[0])
            if start_pos == -1:
                return processed_content
        
        # Find the end position
        end_pos = original_text.find(last_words, start_pos)
        if end_pos == -1:
            # Try with single word
            end_pos = original_text.find(words[-1], start_pos)
            if end_pos == -1:
                return processed_content
        
        # Extract the original text segment
        end_pos += len(last_words)
        original_segment = original_text[start_pos:end_pos]
        
        # If the extracted segment is reasonable, return it
        if len(original_segment) > 0 and len(original_segment) < len(processed_content) * 3:
            return original_segment
        else:
            # Fallback to processed content
            return processed_content
    
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
    
    def _analyze_for_research(self, nodes: List[Dict], edges: List[Dict]) -> Dict[str, Any]:
        """Analyze nodes and edges for research mode synthesis"""
        research_data = {
            'definitive_ideas': [],
            'implementation_code': [],
            'roads_not_taken': [],
            'unique_early_ideas': [],
            'concept_evolution': {}
        }
        
        # Track concept evolution over conversation
        concept_tracker = {}
        
        for i, node in enumerate(nodes):
            content = node.get('content', '')
            node_id = node.get('id', f'node_{i}')
            
            # Extract code blocks
            code_blocks = self._extract_code_blocks(content)
            if code_blocks:
                research_data['implementation_code'].extend([{
                    'node_id': node_id,
                    'code': block,
                    'context': self._get_code_context(content, block)
                } for block in code_blocks])
            
            # Identify definitive statements (latest versions)
            if self._is_definitive_statement(content):
                research_data['definitive_ideas'].append({
                    'node_id': node_id,
                    'idea': content,
                    'confidence': self._calculate_definitiveness_score(content, i, len(nodes))
                })
            
            # Find roads not taken (rejected or superseded ideas)
            if self._identifies_alternative_approach(content):
                research_data['roads_not_taken'].append({
                    'node_id': node_id,
                    'approach': content,
                    'reason': self._extract_rejection_reason(content)
                })
            
            # Capture unique early ideas
            if i < len(nodes) * 0.3 and self._is_unique_idea(content, nodes[i+1:]):
                research_data['unique_early_ideas'].append({
                    'node_id': node_id,
                    'idea': content,
                    'developed': self._was_idea_developed(content, nodes[i+1:])
                })
            
            # Track concept evolution
            concepts = self._extract_key_concepts(content)
            for concept in concepts:
                if concept not in concept_tracker:
                    concept_tracker[concept] = []
                concept_tracker[concept].append({
                    'node_id': node_id,
                    'position': i,
                    'version': content
                })
        
        # Build concept evolution timeline
        for concept, mentions in concept_tracker.items():
            if len(mentions) > 1:
                research_data['concept_evolution'][concept] = {
                    'first_mention': mentions[0],
                    'final_form': mentions[-1],
                    'iterations': len(mentions),
                    'development_path': mentions
                }
        
        return research_data
    
    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract code blocks from content"""
        import re
        # Match code blocks with ``` or indented code
        code_pattern = r'```[\w]*\n(.*?)```|(?:^|\n)((?:    |\t).*(?:\n(?:    |\t).*)*)'
        matches = re.findall(code_pattern, content, re.MULTILINE | re.DOTALL)
        return [match[0] if match[0] else match[1] for match in matches if any(match)]
    
    def _get_code_context(self, content: str, code_block: str) -> str:
        """Get surrounding context for a code block"""
        idx = content.find(code_block)
        if idx == -1:
            return ""
        start = max(0, idx - 200)
        end = min(len(content), idx + len(code_block) + 200)
        return content[start:end].replace(code_block, "[CODE]")
    
    def _is_definitive_statement(self, content: str) -> bool:
        """Check if content represents a definitive/final version of an idea"""
        definitive_markers = [
            'the architecture is', 'final implementation', 'this works',
            'the solution is', 'bottom line', 'key insight', 'core concept',
            'this is how', 'recommended', 'the magic', 'what makes sense'
        ]
        return any(marker in content.lower() for marker in definitive_markers)
    
    def _calculate_definitiveness_score(self, content: str, position: int, total: int) -> float:
        """Calculate how definitive a statement is based on position and language"""
        # Later statements are often more definitive
        position_score = position / total
        
        # Strong language indicators
        strong_indicators = ['definitely', 'absolutely', 'clearly', 'proven', 'works']
        language_score = sum(1 for ind in strong_indicators if ind in content.lower()) / len(strong_indicators)
        
        return (position_score * 0.7 + language_score * 0.3)
    
    def _identifies_alternative_approach(self, content: str) -> bool:
        """Check if content discusses an alternative or rejected approach"""
        alternative_markers = [
            'alternative approach', 'instead of', 'problem:', 'challenge:',
            'this is brittle', 'better approach', 'don\'t', 'avoid',
            'unpredictable', 'unreliable', 'issues with'
        ]
        return any(marker in content.lower() for marker in alternative_markers)
    
    def _extract_rejection_reason(self, content: str) -> str:
        """Extract why an approach was rejected"""
        # Look for explanatory phrases
        reason_patterns = [
            r'because\s+([^.]+)',
            r'due to\s+([^.]+)',
            r'problem:\s*([^.]+)',
            r'issue:\s*([^.]+)',
            r'challenge:\s*([^.]+)'
        ]
        import re
        for pattern in reason_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "Not explicitly stated"
    
    def _is_unique_idea(self, content: str, later_nodes: List[Dict]) -> bool:
        """Check if an idea is unique and not repeated later"""
        # Extract key terms from content
        key_terms = set(word.lower() for word in content.split() if len(word) > 5)
        
        # Check if these terms appear significantly in later nodes
        for node in later_nodes:
            later_content = node.get('content', '').lower()
            overlap = sum(1 for term in key_terms if term in later_content)
            if overlap > len(key_terms) * 0.5:
                return False
        return True
    
    def _was_idea_developed(self, idea: str, later_nodes: List[Dict]) -> bool:
        """Check if an early idea was developed further"""
        idea_keywords = set(word.lower() for word in idea.split() if len(word) > 4)
        
        for node in later_nodes:
            content = node.get('content', '').lower()
            if sum(1 for kw in idea_keywords if kw in content) > 2:
                return True
        return False
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content"""
        # Look for emphasized concepts
        import re
        concepts = []
        
        # Bold/emphasized terms
        bold_pattern = r'\*\*(.*?)\*\*'
        concepts.extend(re.findall(bold_pattern, content))
        
        # Technical terms in backticks
        code_pattern = r'`([^`]+)`'
        concepts.extend(re.findall(code_pattern, content))
        
        # Capitalized multi-word concepts
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        concepts.extend(re.findall(cap_pattern, content))
        
        return list(set(concepts))
    
    def _format_research_output(self, research_data: Dict[str, Any], nodes: List[Dict], original_text: str) -> Dict[str, Any]:
        """Format research analysis into structured output"""
        output = {
            'reassembled_text': self._generate_research_report(research_data),
            'nodes': nodes,
            'research_summary': {
                'total_ideas': len(research_data['definitive_ideas']),
                'code_examples': len(research_data['implementation_code']),
                'alternatives_explored': len(research_data['roads_not_taken']),
                'unique_early_concepts': len(research_data['unique_early_ideas']),
                'concepts_tracked': len(research_data['concept_evolution'])
            },
            'research_data': research_data
        }
        return output
    
    def _generate_research_report(self, research_data: Dict[str, Any]) -> str:
        """Generate a comprehensive research report"""
        report = []
        
        report.append("# Research Mode Analysis Report")
        report.append("*Generated through layered context graph analysis*\n")
        
        # Section 1: Definitive Ideas
        report.append("## 1. Most Definitive Version of Ideas (Latest/Final Forms)")
        report.append("\nThese represent the most refined and conclusive statements from the conversation:\n")
        
        # Sort by confidence score
        definitive_sorted = sorted(research_data['definitive_ideas'], 
                                 key=lambda x: x['confidence'], reverse=True)
        
        for i, idea in enumerate(definitive_sorted[:10], 1):  # Top 10
            report.append(f"### {i}. Idea (Confidence: {idea['confidence']:.2f})")
            report.append(f"{idea['idea'][:500]}...")
            report.append("")
        
        # Section 2: Implementation Details
        report.append("\n## 2. How to Work Them - Code & Implementation")
        report.append("\nPractical implementation examples and code snippets:\n")
        
        for i, code_item in enumerate(research_data['implementation_code'][:15], 1):  # Top 15
            report.append(f"### Code Example {i}")
            report.append(f"**Context**: {code_item['context'][:200]}...")
            report.append("```python")
            report.append(code_item['code'][:1000])  # Limit code length
            report.append("```\n")
        
        # Section 3: Roads Not Taken
        report.append("\n## 3. Roads Not Taken - Rejected Approaches")
        report.append("\nAlternative approaches that were considered but ultimately rejected:\n")
        
        for i, road in enumerate(research_data['roads_not_taken'], 1):
            report.append(f"### Alternative {i}")
            report.append(f"**Approach**: {road['approach'][:300]}...")
            report.append(f"**Reason for rejection**: {road['reason']}")
            report.append("")
        
        # Section 4: Unique Early Ideas
        report.append("\n## 4. Unique Options from Early/Less Developed Ideas")
        report.append("\nInteresting concepts from earlier in the conversation that weren't fully explored:\n")
        
        for i, early_idea in enumerate(research_data['unique_early_ideas'], 1):
            developed_status = "Partially developed" if early_idea['developed'] else "Not developed further"
            report.append(f"### Early Concept {i} ({developed_status})")
            report.append(f"{early_idea['idea'][:400]}...")
            report.append("")
        
        # Section 5: Concept Evolution
        report.append("\n## 5. Concept Evolution Timeline")
        report.append("\nHow key concepts evolved throughout the conversation:\n")
        
        for concept, evolution in list(research_data['concept_evolution'].items())[:10]:
            report.append(f"### '{concept}' Evolution")
            report.append(f"- **First mentioned**: Position {evolution['first_mention']['position']}")
            report.append(f"- **Final form**: Position {evolution['final_form']['position']}")
            report.append(f"- **Total iterations**: {evolution['iterations']}")
            report.append("")
        
        # Summary statistics
        report.append("\n## Summary Statistics")
        report.append(f"- Total definitive ideas extracted: {len(research_data['definitive_ideas'])}")
        report.append(f"- Code implementations found: {len(research_data['implementation_code'])}")
        report.append(f"- Alternative approaches identified: {len(research_data['roads_not_taken'])}")
        report.append(f"- Unique early concepts: {len(research_data['unique_early_ideas'])}")
        report.append(f"- Concepts tracked: {len(research_data['concept_evolution'])}")
        
        return "\n".join(report)
    
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
        """Save processing results with full text preservation"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qwq_layered_results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Create a custom serializer that preserves full text
        def custom_serializer(obj):
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {k: custom_serializer(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [custom_serializer(item) for item in obj]
            else:
                return str(obj)
        
        # Serialize with full content preservation
        serialized_results = custom_serializer(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serialized_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
        
        # Also save comprehensive text output with node list
        text_path = output_path.with_suffix('.txt')
        self._save_text_results(results, text_path)
        
        return output_path
    
    def _save_text_results(self, results: Dict[str, Any], output_path: Path):
        """Save human-readable text results with node list"""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("LAYERED CONTEXT GRAPH PROCESSING RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            # Metadata
            f.write(f"Processing Mode: {results.get('mode', 'N/A')}\n")
            f.write(f"Input Length: {results.get('input_length', 0):,} characters\n")
            f.write(f"Nodes Created: {results.get('nodes', 0)}\n")
            f.write(f"Edges Created: {results.get('edges', 0)}\n")
            f.write(f"Processing Time: {results.get('processing_time', 0):.2f} seconds\n")
            
            if 'metadata' in results:
                meta = results['metadata']
                f.write(f"Model: {meta.get('model', 'N/A')}\n")
                if 'device' in meta:
                    f.write(f"Device: {meta['device']}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            
            # Node List
            if 'output' in results and 'nodes' in results['output']:
                nodes = results['output']['nodes']
                f.write(f"\nNODE LIST ({len(nodes)} nodes)\n")
                f.write("-" * 60 + "\n\n")
                
                for i, node in enumerate(nodes):
                    f.write(f"Node {i} [{node.get('id', f'node_{i}')}]\n")
                    f.write(f"Type: {node.get('segment_type', 'unknown')}\n")
                    
                    # Show content length and preview
                    content = node.get('content', '')
                    f.write(f"Content Length: {len(content)} chars\n")
                    
                    # Show more content (first 500 chars)
                    if len(content) > 500:
                        f.write(f"Content Preview: {content[:500]}...\n")
                    else:
                        f.write(f"Content: {content}\n")
                    
                    # Show formatting info if present
                    if 'formatting' in node and node['formatting']:
                        f.write(f"Formatting: {node['formatting']}\n")
                    
                    # Show importance and other metadata
                    if 'importance' in node:
                        f.write(f"Importance: {node['importance']:.3f}\n")
                    if 'cluster' in node:
                        f.write(f"Cluster: {node['cluster']}\n")
                    
                    f.write("\n")
            
            # Edges Summary with Rich Metadata
            if 'output' in results and 'edges' in results['output']:
                edges = results['output']['edges']
                f.write(f"\nEDGE SUMMARY ({len(edges)} edges)\n")
                f.write("-" * 60 + "\n")
                
                # Count edge types
                edge_types = {}
                for edge in edges:
                    edge_type = edge.get('type', 'unknown')
                    edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
                
                for edge_type, count in sorted(edge_types.items()):
                    f.write(f"  {edge_type}: {count}\n")
                f.write("\n")
                
                # Show detailed edge information for first 10 edges
                f.write("DETAILED EDGE ANALYSIS (First 10 edges)\n")
                f.write("-" * 60 + "\n")
                
                for i, edge in enumerate(edges[:10]):
                    f.write(f"Edge {i+1}: {edge.get('source', 'unknown')} → {edge.get('target', 'unknown')}\n")
                    f.write(f"  Type: {edge.get('type', 'unknown')}\n")
                    f.write(f"  Weight: {edge.get('weight', 0):.3f}\n")
                    
                    # Show metadata if available
                    if 'metadata' in edge:
                        metadata = edge['metadata']
                        f.write(f"  Metadata:\n")
                        for key, value in metadata.items():
                            f.write(f"    {key}: {value}\n")
                    
                    f.write("\n")
                
                if len(edges) > 10:
                    f.write(f"... and {len(edges) - 10} more edges\n\n")
            
            # Full Node Contents (for debugging)
            if 'output' in results and 'nodes' in results['output']:
                nodes = results['output']['nodes']
                f.write("\nFULL NODE CONTENTS\n")
                f.write("=" * 60 + "\n\n")
                
                total_content_length = sum(len(node.get('content', '')) for node in nodes)
                f.write(f"Total content length across all nodes: {total_content_length} chars\n")
                f.write(f"Average content per node: {total_content_length // len(nodes) if nodes else 0} chars\n\n")
                
                for i, node in enumerate(nodes):
                    f.write(f"=== Node {i} Full Content ===\n")
                    
                    # Apply formatting preservation if available
                    content = node.get('content', '[No content]')
                    formatting = node.get('formatting', {})
                    
                    if formatting and hasattr(self, 'graph_reassembler') and hasattr(self.graph_reassembler, '_apply_formatting'):
                        # Use the reassembler to apply formatting
                        formatted_content = self.graph_reassembler._apply_formatting(content, formatting)
                        f.write(formatted_content)
                    else:
                        # Fallback to raw content
                        f.write(content)
                    
                    f.write("\n\n")
            
            # Reconstructed Document
            f.write("=" * 60 + "\n")
            f.write("RECONSTRUCTED DOCUMENT\n")
            f.write("=" * 60 + "\n\n")
            
            if 'output' in results:
                if 'tape2' in results['output']:
                    f.write(results['output']['tape2'])
                elif 'reassembled_text' in results['output']:
                    f.write(results['output']['reassembled_text'])
                else:
                    f.write("[No reconstructed document available]\n")
            else:
                f.write("[No output available]\n")
        
        logger.info(f"Text results saved to {output_path}")
    
    def build_document_progressively(self, graph_data: Dict, strategy: str = 'guided') -> Dict[str, Any]:
        """
        Build document using transformer's progressive generation
        
        Args:
            graph_data: The output from process_text() containing nodes, edges, etc.
            strategy: One of 'linear', 'guided', 'interactive', 'narrative'
            
        Returns:
            Dictionary with generated document and metadata
        """
        if not self.document_builder:
            logger.warning("Transformer document builder not available, falling back to synthesis")
            return self.synthesize_content(graph_data, 'reference')
        
        # Prepare graph data
        doc_input = {
            'nodes': graph_data.get('output', {}).get('nodes', []),
            'edges': graph_data.get('output', {}).get('edges', []),
            'metadata': graph_data.get('metadata', {})
        }
        
        # Generate document progressively
        document = self.document_builder.build_document(doc_input, strategy)
        
        # Save generated document
        if self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"progressive_doc_{strategy}_{timestamp}.md"
            output_path = self.output_dir / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(document)
            
            logger.info(f"Progressive document saved to {output_path}")
        
        return {
            'document': document,
            'strategy': strategy,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'generation_method': 'progressive_transformer',
                'nodes_processed': len(doc_input['nodes']),
                'edges_used': len(doc_input['edges'])
            }
        }
    
    def _process_spectral_hybrid(self, text: str, rules: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Hybrid processing using TorchSpectralProcessor with real attention from QwQ.
        This leverages attention heads and LLMs as primary drivers, with spectral methods as support.
        
        Processing hierarchy:
        1. QwQ model extracts real attention patterns (PRIMARY)
        2. Linguistic programming guides segmentation boundaries (PRIMARY)
        3. Spectral clustering organizes the segments mathematically (SUPPORT)
        4. Graph is built from attention patterns, not just adjacency (PRIMARY)
        
        The spectral methods help organize what the LLM already understands.
        """
        if not self.spectral_processor:
            logger.warning("TorchSpectralProcessor not available, falling back to single-pass")
            return self._process_single_pass(text, rules)
        
        logger.info("Starting spectral-hybrid processing...")
        start_time = datetime.now()
        
        # Determine processing mode
        mode = 'timeline'  # default
        if rules and 'reorganization' in rules:
            mode = rules['reorganization']
        
        # Use linguistic programming by default
        use_linguistic = self.processing_settings.get('use_linguistic_programming', True)
        
        # Process with TorchSpectralProcessor
        try:
            spectral_results = self.spectral_processor.process_conversation(
                text,
                mode=mode,
                use_linguistic_programming=use_linguistic
            )
            
            # Extract nodes and edges from spectral results
            nodes = []
            edges = []
            
            if 'graph' in spectral_results:
                graph = spectral_results['graph']
                
                # Convert spectral nodes to our format
                if 'nodes' in graph:
                    for node in graph['nodes']:
                        nodes.append({
                            'id': node.get('id', f'spectral_node_{node["index"]}'),
                            'content': node.get('content', ''),
                            'cluster': node.get('cluster', 0),
                            'features': node.get('features', {}),
                            'importance': 1.0,  # Can be enhanced with attention scores
                            'spectral_based': True,
                            'segment_type': self._classify_segment_type(node.get('content', ''))
                        })
                
                # Convert spectral edges
                if 'edges' in graph:
                    edges = graph['edges']
            
            # Get reassembled text from spectral results
            reassembled_text = spectral_results.get('reassembled_text', '')
            
            # Create final output structure
            reassembled = {
                'tape2': reassembled_text,
                'nodes': nodes,
                'edges': edges,
                'spectral_metadata': spectral_results.get('spectral_metadata', {})
            }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'mode': 'spectral-hybrid',
                'input_length': len(text),
                'nodes': len(nodes),
                'edges': len(edges),
                'processing_time': processing_time,
                'output': reassembled,
                'metadata': {
                    'model': 'QwQ-32B with Spectral Clustering',
                    'device': str(self.device),
                    'spectral_mode': mode,
                    'linguistic_programming': use_linguistic,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Spectral processing failed: {e}")
            logger.info("Falling back to single-pass processing")
            return self._process_single_pass(text, rules)
    
    def synthesize_content(self, graph_data: Dict, strategy: str = 'executive_summary') -> Dict[str, Any]:
        """
        Synthesize new content from the knowledge graph (Graph → Tape₂)
        
        This is the ACTUAL synthesis that creates new documents, not analysis reports.
        
        Args:
            graph_data: The output from process_text() containing nodes, edges, etc.
            strategy: One of 'executive_summary', 'tutorial', 'reference', 'readme'
            
        Returns:
            Dictionary with synthesized content and metadata
        """
        # Try synthesizers in order of preference
        synthesizer = None
        
        try:
            # First try enhanced coherent synthesizer
            from synthesis.enhanced_tape_synthesizer import EnhancedTapeSynthesizer
            synthesizer = EnhancedTapeSynthesizer()
            logger.info("Using enhanced tape synthesizer for coherent content generation")
        except ImportError:
            pass
        
        if not synthesizer:
            try:
                # Try LLM-based synthesizer
                from synthesis.llm_tape_synthesizer import LLMTapeSynthesizer
                
                # Use QwQ model path if available
                qwq_path = self.config['paths']['project_root'] / 'qwq.gguf'
                model_path = str(qwq_path) if qwq_path.exists() else None
                
                synthesizer = LLMTapeSynthesizer(model_path=model_path)
                logger.info("Using LLM-based tape synthesizer")
                
            except ImportError:
                pass
        
        if not synthesizer:
            # Fall back to original synthesizer
            logger.warning("Advanced synthesizers not available, using basic synthesis")
            from synthesis.tape_synthesizer import TapeSynthesizer
            synthesizer = TapeSynthesizer()
        
        # Prepare graph data
        synthesis_input = {
            'nodes': graph_data.get('output', {}).get('nodes', []),
            'edges': graph_data.get('output', {}).get('edges', []),
            'original_text': graph_data.get('original_text', '')
        }
        
        # Generate synthesized content
        result = synthesizer.synthesize(synthesis_input, strategy)
        
        # Save synthesized content
        if self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"synthesized_{strategy}_{timestamp}.md"
            output_path = self.output_dir / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['tape2'])
            
            logger.info(f"Synthesized content saved to {output_path}")
            result['output_path'] = str(output_path)
        
        return result
    
    def process_conversation(self, text: str, mode: str = 'timeline', use_spectral: bool = None) -> Dict[str, Any]:
        """
        Process conversation with specific reconstruction mode.
        
        Args:
            text: Conversation transcript
            mode: One of 'timeline', 'speaker', 'evolution', 'current_state', 'research'
            use_spectral: Whether to use spectral processing (None = auto-decide based on GPU availability)
        
        Returns:
            Processed conversation in the requested format
        """
        logger.info(f"Processing conversation in '{mode}' mode...")
        
        # Decide whether to use spectral processing
        if use_spectral is None:
            use_spectral = self.spectral_processor is not None and torch.cuda.is_available()
        
        # If spectral processing is requested and available, use it directly
        if use_spectral and self.spectral_processor:
            logger.info("Using TorchSpectralProcessor for conversation processing")
            return self.spectral_processor.process_conversation(text, mode=mode, use_linguistic_programming=True)
        
        # Otherwise, use traditional processing
        # Enable conversation-specific disassembly rules
        if hasattr(self.partition_manager, 'disassembly_rules'):
            self.partition_manager.disassembly_rules['conversation_boundaries'] = True
        
        # Process with conversation-specific rules
        rules = {
            'segmentation': 'conversation_boundaries',
            'reorganization': mode
        }
        
        # Run the processing pipeline
        result = self.process_text(text, rules)
        
        # Apply conversation-specific reassembly based on mode
        if 'output' in result:
            nodes = result['output'].get('nodes', [])
            edges = result['output'].get('edges', [])
            
            # Use conversation edge analyzer to enhance edges
            from graph.conversation_edge_types import ConversationEdgeAnalyzer
            edge_analyzer = ConversationEdgeAnalyzer(self.attention_extractor)
            
            # Enhance edges with conversation-specific types
            enhanced_edges = edge_analyzer.create_conversation_edges(nodes, use_attention=True)
            
            # Apply the specific reassembly method or research analysis
            if mode == 'research':
                # Use research analysis instead of reassembly
                research_analysis = self._analyze_for_research(nodes, enhanced_edges)
                conversation_output = self._format_research_output(research_analysis, nodes, text)
            elif hasattr(self.graph_reassembler, f'reassemble_by_{mode}'):
                reassembly_method = getattr(self.graph_reassembler, f'reassemble_by_{mode}')
                conversation_output = reassembly_method(nodes, enhanced_edges, text)
            else:
                # Fallback to standard reassembly
                conversation_output = self.graph_reassembler.reassemble_graph(nodes, enhanced_edges, text)
            
            # Merge with main result
            result['conversation_output'] = conversation_output
            result['conversation_mode'] = mode
            result['enhanced_edges'] = len(enhanced_edges)
        
        return result


# Import demo content loader
from load_demo_content import get_demo_content


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
  
  # Generate synthesis (Tape₂)
  python master_processor_full.py --demo technical --synthesize executive_summary
  python master_processor_full.py --input doc.txt --synthesize tutorial
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['single-pass', 'multi-round', 'language-guided', 'spectral-hybrid'],
        default='single-pass',
        help='Processing mode (spectral-hybrid uses GPU-accelerated attention-based processing)'
    )
    
    # Input/Output
    parser.add_argument('--input', '-i', help='Input text file path')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--demo', choices=list(DEMO_CONFIGS.keys()) + ['conversation', 'layered_context', 'layered_context_file'],
                       help='Use demo content')
    
    # Processing options
    parser.add_argument('--rules', choices=RULE_SETS.keys(),
                       help='Predefined rule set')
    parser.add_argument('--conversation-mode', 
                       choices=['timeline', 'speaker', 'evolution', 'current_state', 'research'],
                       default='timeline',
                       help='Conversation reassembly mode (when processing conversations)')
    parser.add_argument('--synthesize',
                       choices=['executive_summary', 'tutorial', 'reference', 'readme'],
                       help='Synthesize new content type from the graph (Tape₂ generation)')
    parser.add_argument('--progressive',
                       choices=['linear', 'guided', 'interactive', 'narrative'],
                       help='Build document progressively using transformer generation')
    
    # Advanced options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--preserve-content', action='store_true',
                       help='Use content-preserving settings for better text retention')
    
    # Spectral processing options
    parser.add_argument('--gpu', action='store_true',
                       help='Force GPU usage for spectral processing')
    parser.add_argument('--use-spectral', action='store_true',
                       help='Use spectral processing for conversations (auto-enabled with --mode spectral-hybrid)')
    parser.add_argument('--no-linguistic', action='store_true',
                       help='Disable linguistic programming in spectral mode')
    
    # Graph configuration options
    parser.add_argument('--graph-preset',
                       choices=['default', 'conversation', 'code', 'research', 'spectral'],
                       help='Use a preset graph configuration')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(mode=args.mode, model_type='ollama')
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Override output directory
    if args.output:
        config['paths']['results_dir'] = Path(args.output)
    
    # Add spectral processing settings
    if args.gpu:
        config['device'] = 'cuda'
    if args.use_spectral or args.mode == 'spectral-hybrid':
        config['use_spectral'] = True
    if args.no_linguistic:
        config['processing_settings']['use_linguistic_programming'] = False
    
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
    
    # Get graph configuration if specified
    graph_config = None
    if GraphConfig and get_graph_config:
        if args.graph_preset:
            graph_config = get_graph_config(args.graph_preset)
        else:
            # Auto-select based on content type
            if args.demo == 'conversation' or ('Speaker' in text and ':' in text):
                graph_config = get_graph_config('conversation')
            elif args.demo == 'code' or '```' in text or 'def ' in text:
                graph_config = get_graph_config('code')
            else:
                graph_config = get_graph_config('default')
        
        # Apply formatting preservation as default behavior
        if graph_config:
            graph_config.reconstruction.preserve_original_formatting = True
            graph_config.reconstruction.preserve_indentation = True
            graph_config.reconstruction.preserve_empty_lines = True
            graph_config.disassembly.preserve_whitespace = True
            graph_config.disassembly.preserve_line_breaks = True
    
    # Process
    try:
        processor = FullMasterProcessor(config, graph_config)
        logger.info(f"Starting {args.mode} processing with QwQ...")
        
        # Check if this is a conversation
        if args.demo == 'conversation' or ('Speaker' in text and ':' in text):
            logger.info(f"Detected conversation - using '{args.conversation_mode}' reassembly mode")
            results = processor.process_conversation(
                text, 
                mode=args.conversation_mode,
                use_spectral=args.use_spectral
            )
        else:
            results = processor.process_text(text, rules)
        
        # Save results
        output_path = processor.save_results(results)
        
        # If synthesis requested, generate Tape₂
        # Progressive document building
        if args.progressive:
            print(f"\n🔄 Building document progressively using {args.progressive} strategy...")
            progressive_doc = processor.build_document_progressively(results, args.progressive)
            
            if 'document' in progressive_doc:
                print(f"✓ Progressive document generated with {progressive_doc['metadata']['nodes_processed']} nodes")
                print(f"  Strategy: {progressive_doc['strategy']}")
                print(f"  Length: {len(progressive_doc['document'])} characters")
        
        # Traditional synthesis
        elif args.synthesize:
            print(f"\n🔄 Generating {args.synthesize} synthesis...")
            synthesized = processor.synthesize_content(results, args.synthesize)
            
            # Save synthesized content
            synth_filename = f"synthesized_{args.synthesize}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            synth_path = processor.output_dir / synth_filename
            with open(synth_path, 'w', encoding='utf-8') as f:
                f.write(synthesized.get('tape2', ''))
            
            print(f"✅ Synthesized content saved to: {synth_path}")
            results['synthesis'] = {
                'type': args.synthesize,
                'output_path': str(synth_path),
                'length': len(synthesized.get('tape2', ''))
            }
        
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
        
        # Show synthesis results if generated
        if 'synthesis' in results:
            print(f"\nSynthesis Generated:")
            print(f"  Type: {results['synthesis']['type']}")
            print(f"  Output: {results['synthesis']['output_path']}")
            print(f"  Length: {results['synthesis']['length']:,} characters")
        
        # Show conversation-specific results if available
        if 'conversation_output' in results:
            conv_output = results['conversation_output']
            print(f"\nConversation Analysis:")
            print(f"  Mode: {results['conversation_mode']}")
            print(f"  Enhanced edges: {results['enhanced_edges']}")
            
            # Research mode specific output
            if results['conversation_mode'] == 'research' and 'research_summary' in conv_output:
                summary = conv_output['research_summary']
                print(f"\nResearch Analysis Summary:")
                print(f"  Definitive ideas found: {summary['total_ideas']}")
                print(f"  Code implementations: {summary['code_examples']}")
                print(f"  Alternative approaches: {summary['alternatives_explored']}")
                print(f"  Unique early concepts: {summary['unique_early_concepts']}")
                print(f"  Concepts tracked: {summary['concepts_tracked']}")
            elif 'speaker_count' in conv_output:
                print(f"  Speakers found: {conv_output['speaker_count']}")
            if 'concept_chains' in conv_output:
                print(f"  Concept evolution chains: {conv_output['concept_chains']}")
            if 'topics_resolved' in conv_output:
                print(f"  Topics analyzed: {conv_output['topics_resolved']}")
        
        print("="*60)
        
        # Show output files
        print(f"\n📁 Output saved to:")
        print(f"  JSON: {output_path}")
        print(f"  Text: {output_path.with_suffix('.txt')}")
        
        # If no synthesis was requested, show available options
        if not args.synthesize and not args.progressive:
            print("\n💡 Tip: Generate synthesized content with:")
            print("  --synthesize executive_summary  # High-level overview")
            print("  --synthesize tutorial          # Step-by-step guide")
            print("  --synthesize reference         # Technical reference")
            print("  --synthesize readme           # Project documentation")
            print("\n  Or build progressively with:")
            print("  --progressive linear      # Sequential generation")
            print("  --progressive guided      # Context-aware generation")
            print("  --progressive narrative   # Story-like flow")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()