#!/usr/bin/env python3
"""
Simplified Master Processor for Layered Context Graph
This version avoids the problematic imports
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add src directory to path
project_root = Path(__file__).parent.resolve()
src_path = project_root / "layered-context-graph" / "src"
sys.path.insert(0, str(src_path))

# Import config
from master_config import get_config, get_rule_set, DEMO_CONFIGS, RULE_SETS

# Import only the modules that work
from models.context_window import ContextWindow
from models.attention_extractor import EnhancedAttentionExtractor
from models.instruction_seeder import InstructionSeeder
from partitioning.partition_manager import PartitionManager
from graph.graph_reassembler import GraphReassembler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleMasterProcessor:
    """Simplified processor that works with available modules"""
    
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
        """Initialize processing components"""
        logger.info(f"Initializing components for {self.mode} mode with {self.model_type} model")
        
        self.context_window = ContextWindow(
            size=self.processing_settings.get('window_size', 2000)
        )
        self.seeder = InstructionSeeder()
        
        model_name = self.model_config.get('default_model', 'distilbert-base-uncased')
        
        try:
            # EnhancedAttentionExtractor only takes model_path parameter
            if self.model_type == 'ollama':
                # For ollama, pass the model path
                self.attention_extractor = EnhancedAttentionExtractor(model_name)
            else:
                # For transformer models, don't pass any parameter (will use default)
                self.attention_extractor = EnhancedAttentionExtractor()
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            # Try with default
            self.attention_extractor = EnhancedAttentionExtractor()
        
        self.partition_manager = PartitionManager(
            overlap_ratio=self.processing_settings.get('overlap_ratio', 0.1),
            min_chunk_size=self.processing_settings.get('min_chunk_size', 100)
        )
        self.graph_reassembler = GraphReassembler(
            similarity_threshold=self.processing_settings.get('similarity_threshold', 0.95)
        )
    
    def process_text(self, text: str, rules: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process text using available components"""
        logger.info(f"Starting {self.mode} processing...")
        
        start_time = datetime.now()
        
        # Step 1: Create semantic windows
        logger.info("Creating semantic windows...")
        semantic_windows = self.context_window.create_window(text)
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
        partitions = self.partition_manager.create_partitions(semantic_windows)
        
        # Step 5: Reassemble
        logger.info("Reassembling graph...")
        # GraphReassembler expects nodes and edges, so we need to convert partitions
        nodes = [{'id': f'partition_{i}', 'content': p} for i, p in enumerate(partitions)]
        edges = []  # No edges for simple processing
        reassembled = self.graph_reassembler.reassemble_graph(nodes, edges, text)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'mode': self.mode,
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
        
        logger.info(f"Processing completed in {processing_time:.2f}s")
        return result
    
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
    """Get demo content"""
    demo_map = {
        'transcript': """
        Speaker A: Let's discuss the new architecture for our system.
        Speaker B: I think we should use a microservices approach.
        Speaker A: That makes sense. We'll need to consider service discovery.
        Speaker B: And don't forget about load balancing and fault tolerance.
        """,
        'technical': """
        ## System Architecture
        
        The system uses a layered architecture with the following components:
        
        ```python
        class DataProcessor:
            def process(self, data):
                # Process the data
                return processed_data
        ```
        
        ### Key Features
        - Scalable processing
        - Fault tolerance
        - Real-time analytics
        """,
        'simple': """
        This is a simple example text that demonstrates basic processing.
        It contains multiple sentences and paragraphs.
        
        The processor will analyze this text and create a layered context graph.
        """
    }
    return demo_map.get(demo_type, "No demo content found.")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Simplified Layered Context Graph Processor',
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        logging.getLogger().setLevel(logging.DEBUG)
    
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
        processor = SimpleMasterProcessor(config)
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
        print("="*60)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()