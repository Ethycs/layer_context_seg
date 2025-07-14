#!/usr/bin/env python3
"""
Layered Context Graph Master Script
==================================

Unified processing script for layered context graphs with multi-round annotation support.
Combines all functionality from the various test scripts into a single configurable tool.

Usage:
    python master_processor.py --mode single-pass --input text.txt --output results/
    python master_processor.py --mode multi-round --model ollama --rules custom
    python master_processor.py --demo transcript --preserve-code
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add src directory to path for robust execution
project_root = Path(__file__).parent.resolve()
src_path = project_root / "layered-context-graph" / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Unified config import
from master_config import get_config, get_rule_set, DEMO_CONFIGS, RULE_SETS

try:
    # Core imports
    from models.context_window import ContextWindow
    from models.attention_extractor import EnhancedAttentionExtractor
    from models.instruction_seeder import InstructionSeeder
    from partitioning.partition_manager import PartitionManager
    from graph.graph_reassembler import GraphReassembler
    from processor.language_guided_processor import LanguageGuidedProcessor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the correct directory with all dependencies installed")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MasterProcessor:
    """Unified processor for all layered context graph operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode = self.config['mode']
        self.model_type = self.config['model_type']
        self.model_config = self.config['model_config']
        self.processing_settings = self.config['processing_settings']
        self.output_dir = self.config['paths']['results_dir']
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components based on mode
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize processing components based on configuration"""
        logger.info(f"Initializing components for {self.mode} mode with {self.model_type} model")
        
        if self.mode == 'multi-round':
            self._setup_multi_round()
        elif self.mode == 'language-guided':
            self._setup_language_guided()
        else:  # single-pass
            self._setup_single_pass()
    
    def _setup_single_pass(self):
        """Setup for single-pass processing"""
        self.context_window = ContextWindow(
            size=self.processing_settings.get('window_size', 2000)
        )
        self.seeder = InstructionSeeder()
        
        model_name = self.model_config.get('default_model', 'distilbert-base-uncased')
        
        try:
            self.attention_extractor = EnhancedAttentionExtractor(
                model_name, model_type=self.model_type
            )
        except Exception as e:
            logger.warning(f"Failed to load preferred model, falling back to distilbert: {e}")
            self.attention_extractor = EnhancedAttentionExtractor(
                "distilbert-base-uncased", model_type="transformer"
            )
        
        self.partition_manager = PartitionManager(
            overlap_ratio=self.processing_settings.get('overlap_ratio', 0.1),
            min_chunk_size=self.processing_settings.get('min_chunk_size', 100)
        )
        self.graph_reassembler = GraphReassembler(
            similarity_threshold=self.processing_settings.get('similarity_threshold', 0.95)
        )
    
    def _setup_multi_round(self):
        """Setup for multi-round annotation processing"""
        self.multi_round_config = self.config.get('multi_round', {})
        
        # Base components
        self._setup_single_pass()
        logger.info("Multi-round annotation components initialized")
    
    def _setup_language_guided(self):
        """Setup for language-guided processing"""
        self.processor = LanguageGuidedProcessor(
            model_source=self.model_config.get('default_model'),
            model_type=self.model_type
        )
    
    def process_text(self, text: str, rules: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process text using the configured mode"""
        
        if self.mode == 'language-guided':
            return self._process_language_guided(text, rules)
        elif self.mode == 'multi-round':
            return self._process_multi_round(text, rules)
        else:
            return self._process_single_pass(text, rules)
    
    def _process_single_pass(self, text: str, rules: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Single-pass processing"""
        logger.info("Starting single-pass processing...")
        
        start_time = datetime.now()
        
        # Step 1: Create semantic windows
        logger.info("Creating semantic windows...")
        semantic_windows = self.context_window.create_semantic_windows(text)
        logger.info(f"Created {len(semantic_windows)} semantic windows")
        
        # Step 2: Seed with instructions if rules provided
        if rules:
            logger.info("Applying custom rules...")
            seeded_text = self.seeder.seed_instructions(text, rules)
        else:
            seeded_text = text
        
        # Step 3: Extract attention patterns
        logger.info("Extracting attention patterns...")
        attention_patterns = self.attention_extractor.extract_attention(seeded_text)
        
        # Step 4: Create partitions
        logger.info("Creating partitions...")
        partitions = self.partition_manager.create_partitions(
            semantic_windows, attention_patterns
        )
        
        # Step 5: Reassemble
        logger.info("Reassembling graph...")
        reassembled = self.graph_reassembler.reassemble(partitions, text)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'mode': 'single-pass',
            'input_length': len(text),
            'semantic_windows': len(semantic_windows),
            'partitions': len(partitions),
            'processing_time': processing_time,
            'output': reassembled,
            'metadata': {
                'model_type': self.model_type,
                'model_name': self.model_config.get('default_model'),
                'rules_applied': bool(rules),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info(f"Single-pass processing completed in {processing_time:.2f}s")
        return result
    
    def _process_multi_round(self, text: str, rules: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Multi-round annotation processing"""
        logger.info("Starting multi-round processing...")
        
        start_time = datetime.now()
        
        # Step 1: Create base graph (same as single-pass initial steps)
        logger.info("Creating base graph...")
        base_result = self._process_single_pass(text, rules)
        
        # Step 2: Apply multi-round annotations
        logger.info("Applying multi-round annotations...")
        
        # Simulate multi-round annotation layers
        annotations = {}
        
        for round_name, round_config in self.multi_round_config['analysis_rounds'].items():
            logger.info(f"Applying {round_name} analysis...")
            
            # Simulate layer-specific analysis
            layer_annotations = self._simulate_layer_analysis(
                text, round_name, round_config
            )
            annotations[round_name] = layer_annotations
        
        # Step 3: Cross-layer synthesis
        logger.info("Performing cross-layer synthesis...")
        synthesized_insights = self._synthesize_layers(annotations)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'mode': 'multi-round',
            'base_result': base_result,
            'layer_annotations': annotations,
            'synthesized_insights': synthesized_insights,
            'processing_time': processing_time,
            'metadata': {
                'rounds_applied': list(annotations.keys()),
                'synthesis_config': self.multi_round_config['synthesis'],
                'timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info(f"Multi-round processing completed in {processing_time:.2f}s")
        return result
    
    def _process_language_guided(self, text: str, rules: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Language-guided processing"""
        logger.info("Starting language-guided processing...")
        
        start_time = datetime.now()
        
        # Use the language guided processor
        result = self.processor.process_with_natural_language_rules(
            text=text,
            segmentation_rule=rules.get('segmentation') if rules else None,
            reorganization_rule=rules.get('reorganization') if rules else None
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result.update({
            'mode': 'language-guided',
            'processing_time': processing_time,
            'metadata': {
                'model_type': self.model_type,
                'model_name': self.model_config.get('default_model'),
                'timestamp': datetime.now().isoformat()
            }
        })
        
        logger.info(f"Language-guided processing completed in {processing_time:.2f}s")
        return result
    
    def _simulate_layer_analysis(self, text: str, layer_name: str, config: Dict) -> Dict[str, Any]:
        """Simulate layer-specific analysis (placeholder for actual implementation)"""
        
        # This is a simplified simulation - in real implementation,
        # you would use the specified models and extract actual features
        
        analysis = {
            'layer_type': layer_name,
            'model_used': config['model'],
            'features_extracted': config['features'],
            'weight': config['weight'],
            'analysis_results': {
                'node_count': len(text.split()) // 10,  # Simulated
                'complexity_score': min(len(text) / 1000, 1.0),  # Simulated
                'confidence': 0.8  # Simulated
            }
        }
        
        return analysis
    
    def _synthesize_layers(self, annotations: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize insights across multiple annotation layers"""
        
        synthesis = {
            'confidence_score': sum(
                layer['analysis_results']['confidence'] * layer['weight']
                for layer in annotations.values()
            ),
            'complexity_aggregate': sum(
                layer['analysis_results']['complexity_score']
                for layer in annotations.values()
            ) / len(annotations),
            'cross_layer_patterns': [
                f"Correlation between {layer1} and {layer2}"
                for layer1 in annotations.keys()
                for layer2 in annotations.keys()
                if layer1 != layer2
            ]
        }
        
        return synthesis
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> Path:
        """Save processing results to file"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"layered_context_results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
        return output_path

def get_demo_content(demo_type: str) -> str:
    """Get demo content from master_config"""
    # This is a simplified representation. A real implementation might load from a file.
    demo_map = {
        'transcript': """
        Meeting Discussion on AI Development Strategy...
        """,
        'technical': """
        # Layered Context Graph Architecture...
        """,
        'simple': """
        This is a simple test document...
        """
    }
    return demo_map.get(demo_type, "No demo content found.")

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Layered Context Graph Master Processor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-pass processing with transformer model
  python master_processor.py --mode single-pass --demo transcript
  
  # Multi-round annotation with Ollama model
  python master_processor.py --mode multi-round --model-type ollama
  
  # Language-guided processing with custom rules
  python master_processor.py --mode language-guided --rules academic_paper
  
  # Process custom file
  python master_processor.py --input my_document.txt --output results/ --mode multi-round
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', 
        choices=['single-pass', 'multi-round', 'language-guided'],
        default='single-pass',
        help='Processing mode to use'
    )
    
    # Input/Output
    parser.add_argument('--input', '-i', help='Input text file path')
    parser.add_argument('--output', '-o', help='Output directory (overrides config)')
    parser.add_argument('--demo', choices=DEMO_CONFIGS.keys(), 
                       help='Use demo content instead of input file')
    
    # Model configuration
    parser.add_argument('--model-type', choices=['transformer', 'ollama'], 
                       default='transformer', help='Model type to use')
    
    # Processing options
    parser.add_argument('--rules', choices=RULE_SETS.keys(),
                       help='Predefined rule set to use')
    
    # Advanced options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Get base configuration
    config = get_config(mode=args.mode, model_type=args.model_type)
    
    # Configure logging level
    if args.verbose:
        config['logging']['level'] = 'DEBUG'
    logging.basicConfig(
        level=config['logging']['level'],
        format=config['logging']['format']
    )
    
    # Override output dir if provided
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
        processor = MasterProcessor(config)
        logger.info(f"Starting {args.mode} processing...")
        
        results = processor.process_text(text, rules)
        
        # Save results
        output_path = processor.save_results(results)
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Mode: {results['mode']}")
        print(f"Input length: {results.get('input_length', len(text))} characters")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        print(f"Results saved to: {output_path}")
        
        if args.mode == 'multi-round':
            print(f"Annotation layers: {list(results['layer_annotations'].keys())}")
            print(f"Synthesis confidence: {results['synthesized_insights']['confidence_score']:.3f}")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
