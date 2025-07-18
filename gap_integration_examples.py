#!/usr/bin/env python3
"""
GAP Integration Examples
========================
Examples showing how to use the new dual-level processing capabilities
with the enhanced QwQ model and partition manager.
"""

import logging
import numpy as np
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_dual_level_processing():
    """
    Basic example of using dual-level attention extraction.
    """
    print("\n=== Basic Dual-Level Processing Example ===\n")
    
    # Import enhanced model
    from qwq_model_merge import QwQModelDualLevel
    
    # Initialize model with dual-level enabled
    model = QwQModelDualLevel(
        model_path="./QwQ-32B",
        enable_dual_level=True
    )
    
    # Example text with clear segments
    text = """
    Machine learning is a subset of artificial intelligence. It focuses on 
    building systems that learn from data. Deep learning is a subset of 
    machine learning. It uses neural networks with multiple layers.
    
    Natural language processing is another AI field. It deals with understanding
    human language. Recent advances use transformer models. These models have
    revolutionized the field.
    """
    
    # Define segments (could come from PartitionManager)
    segments = [
        {
            'id': '0',
            'content': 'Machine learning is a subset of artificial intelligence. It focuses on building systems that learn from data.',
            'start_pos': 0,
            'end_pos': 108
        },
        {
            'id': '1', 
            'content': 'Deep learning is a subset of machine learning. It uses neural networks with multiple layers.',
            'start_pos': 109,
            'end_pos': 201
        },
        {
            'id': '2',
            'content': 'Natural language processing is another AI field. It deals with understanding human language.',
            'start_pos': 202,
            'end_pos': 293
        },
        {
            'id': '3',
            'content': 'Recent advances use transformer models. These models have revolutionized the field.',
            'start_pos': 294,
            'end_pos': 377
        }
    ]
    
    # Create simple adjacency matrix (segments 0-1 and 2-3 are connected)
    adjacency = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])
    
    # Extract dual-level attention
    results = model.extract_dual_level_attention(
        text=text,
        segments=segments,
        adjacency_matrix=adjacency,
        window_size=256,  # Smaller window for example
        rank_ratio=0.2
    )
    
    # Display results
    print(f"Processed {len(results)} windows")
    for i, window_result in enumerate(results):
        print(f"\nWindow {i}:")
        print(f"  - Tokens: {window_result['sequence_length']}")
        print(f"  - Has dual-level: {window_result.get('dual_level', False)}")
        
        if 'layers' in window_result:
            layer = window_result['layers'][0]  # First layer
            print(f"  - Token attention shape: {layer['token_attention'].shape}")
            if layer['node_attention'] is not None:
                print(f"  - Node attention shape: {layer['node_attention'].shape}")


def example_with_partition_manager():
    """
    Example using PartitionManager with dual-level processing.
    """
    print("\n=== Partition Manager with Dual-Level Example ===\n")
    
    # Import components
    from layered_context_graph.src.partitioning.partition_manager import PartitionManager
    from qwq_model_merge import QwQModelDualLevel
    
    # Create enhanced partition manager
    class DualLevelPartitionManager(PartitionManager):
        """Extended partition manager with dual-level support."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._dual_segmenter = None
        
        @property
        def dual_segmenter(self):
            if self._dual_segmenter is None:
                self._dual_segmenter = QwQModelDualLevel(
                    self.qwq_model_path,
                    enable_dual_level=True
                )
            return self._dual_segmenter
        
        def partition_dual_level(self, text: str, k_rules: List[str]):
            """Partition with dual-level analysis."""
            # First, standard partitioning
            self.partition(text, k_rules)
            
            # Convert segments for dual-level
            segments_list = []
            for seg_id, segment in self.segments.items():
                segments_list.append({
                    'id': seg_id,
                    'content': segment.content,
                    'start_pos': segment.start_pos,
                    'end_pos': segment.end_pos
                })
            
            # Extract adjacency from graph
            adjacency = nx.to_numpy_array(self.graph)
            
            # Run dual-level analysis
            logger.info("Running dual-level attention analysis...")
            dual_results = self.dual_segmenter.extract_dual_level_attention(
                text=text,
                segments=segments_list,
                adjacency_matrix=adjacency
            )
            
            # Store results
            self.dual_attention_results = dual_results
            logger.info(f"Dual-level analysis complete: {len(dual_results)} windows")
            
            return dual_results
    
    # Initialize manager
    pm = DualLevelPartitionManager()
    
    # Example document
    document = """
    Introduction to Neural Networks
    
    Neural networks are computing systems inspired by biological neural networks.
    They consist of interconnected nodes or neurons. Each connection can transmit
    signals between neurons.
    
    Architecture Overview
    
    A typical neural network has an input layer, hidden layers, and an output layer.
    The input layer receives data. Hidden layers process the information. The output
    layer produces the final result.
    
    Training Process
    
    Training involves adjusting the connection weights. This is done using
    backpropagation. The network learns patterns from training data.
    """
    
    # Define segmentation rules
    rules = [
        "Split by major sections (look for headers)",
        "Split paragraphs within sections"
    ]
    
    # Run dual-level partitioning
    dual_results = pm.partition_dual_level(document, rules)
    
    # Display graph structure
    print(f"Created graph with {len(pm.segments)} nodes")
    print(f"Graph edges: {pm.graph.number_of_edges()}")
    
    # Show segment relationships
    for edge in pm.graph.edges(data=True):
        print(f"  {edge[0][:8]}... -> {edge[1][:8]}... [{edge[2].get('type', 'unknown')}]")


def example_attention_calculator_usage():
    """
    Example using the enhanced attention calculator.
    """
    print("\n=== Dual-Level Attention Calculator Example ===\n")
    
    from attention_calculator_merge import DualLevelAttentionCalculator
    from qwq_model_merge import QwQModelDualLevel
    
    # Initialize calculator
    calculator = DualLevelAttentionCalculator(
        rank_ratio=0.1,
        boundary_threshold=0.3,
        node_attention_weight=0.6
    )
    
    # Initialize model
    model = QwQModelDualLevel("./QwQ-32B", enable_dual_level=True)
    
    # Example text
    text = """
    The water cycle describes how water moves on Earth. Water evaporates from 
    oceans and lakes. It forms clouds in the atmosphere. Then it falls as rain
    or snow.
    
    Plants also play a role through transpiration. They release water vapor 
    through their leaves. This adds moisture to the air. The process continues
    in an endless cycle.
    """
    
    # Simple segments
    segments = [
        {'id': '0', 'content': 'The water cycle describes...', 'start_pos': 0, 'end_pos': 100},
        {'id': '1', 'content': 'Plants also play a role...', 'start_pos': 101, 'end_pos': 200}
    ]
    
    # Extract with calculator
    results = model.extract_dual_level_attention(
        text=text,
        segments=segments,
        calculator=calculator
    )
    
    # Get comprehensive results
    final_results = calculator.get_dual_level_results()
    
    print("Dual-Level Analysis Results:")
    print(f"  - Suggested segments: {len(final_results['suggested_segments'])}")
    print(f"  - Segment boundaries: {len(final_results['segment_boundaries'])}")
    
    if 'dual_level_analysis' in final_results:
        dual = final_results['dual_level_analysis']
        print("\nNode-Level Patterns:")
        print(f"  - Mean clustering: {dual['node_patterns'].get('mean_clustering', 0):.3f}")
        print(f"  - Graph density: {dual['node_patterns'].get('mean_density', 0):.3f}")
        
        print("\nSegment Cohesion:")
        print(f"  - Mean cohesion: {dual['segment_cohesion'].get('mean_cohesion', 0):.3f}")
        print(f"  - Cohesion variance: {dual['segment_cohesion'].get('cohesion_variance', 0):.3f}")


def example_hierarchical_processing():
    """
    Example of hierarchical dual-level processing.
    """
    print("\n=== Hierarchical Dual-Level Processing Example ===\n")
    
    from token_segment_mapper import HierarchicalTokenMapper
    from dual_level_processor import create_hierarchical_dual_processor
    
    # Create hierarchical processors
    processors = create_hierarchical_dual_processor(num_levels=3)
    
    # Example hierarchical segments
    hierarchy = {
        0: [  # Top level - chapters
            {'id': 'ch1', 'content': 'Chapter 1...', 'tokens': list(range(0, 100))},
            {'id': 'ch2', 'content': 'Chapter 2...', 'tokens': list(range(100, 200))}
        ],
        1: [  # Mid level - sections
            {'id': 'sec1.1', 'content': 'Section 1.1...', 'tokens': list(range(0, 50))},
            {'id': 'sec1.2', 'content': 'Section 1.2...', 'tokens': list(range(50, 100))},
            {'id': 'sec2.1', 'content': 'Section 2.1...', 'tokens': list(range(100, 150))},
            {'id': 'sec2.2', 'content': 'Section 2.2...', 'tokens': list(range(150, 200))}
        ],
        2: [  # Bottom level - paragraphs
            {'id': 'p1', 'content': 'Paragraph 1...', 'tokens': list(range(0, 25))},
            {'id': 'p2', 'content': 'Paragraph 2...', 'tokens': list(range(25, 50))},
            # ... more paragraphs
        ]
    }
    
    print(f"Hierarchical structure:")
    for level, segments in hierarchy.items():
        print(f"  Level {level}: {len(segments)} segments")
    
    # Create mapper
    mapper = HierarchicalTokenMapper(hierarchy_levels=3)
    
    # Example: Find multi-level mapping for token 75
    token_idx = 75
    multi_level = {
        0: 0,  # Chapter 1
        1: 1,  # Section 1.2
        2: 3   # Paragraph 4
    }
    
    print(f"\nToken {token_idx} belongs to:")
    for level, seg_idx in multi_level.items():
        print(f"  Level {level}: Segment {seg_idx}")


def example_performance_comparison():
    """
    Compare performance with and without dual-level processing.
    """
    print("\n=== Performance Comparison Example ===\n")
    
    import time
    from qwq_model_merge import QwQModelDualLevel
    
    # Test text
    test_text = " ".join(["This is a test sentence."] * 100)  # Repeat for length
    
    # Test segments
    test_segments = [
        {'id': str(i), 'content': f'Segment {i}', 'start_pos': i*100, 'end_pos': (i+1)*100}
        for i in range(10)
    ]
    
    # Model with dual-level disabled
    model_standard = QwQModelDualLevel("./QwQ-32B", enable_dual_level=False)
    
    # Model with dual-level enabled
    model_dual = QwQModelDualLevel("./QwQ-32B", enable_dual_level=True)
    
    # Time standard extraction
    start = time.time()
    standard_results = model_standard.extract_attention(test_text, window_size=256)
    standard_time = time.time() - start
    
    # Time dual-level extraction
    start = time.time()
    dual_results = model_dual.extract_dual_level_attention(
        test_text, test_segments, window_size=256
    )
    dual_time = time.time() - start
    
    print(f"Standard extraction time: {standard_time:.3f}s")
    print(f"Dual-level extraction time: {dual_time:.3f}s")
    print(f"Overhead: {(dual_time/standard_time - 1)*100:.1f}%")
    
    print(f"\nAdditional information from dual-level:")
    print(f"  - Node-level patterns: ✓")
    print(f"  - Cross-level attention: ✓")
    print(f"  - Segment cohesion: ✓")
    print(f"  - Graph-aware boundaries: ✓")


def main():
    """Run all examples."""
    print("GAP Integration Examples")
    print("=" * 50)
    
    # Note: These examples assume models are available
    # In practice, you'd need to handle model loading
    
    try:
        example_basic_dual_level_processing()
    except Exception as e:
        print(f"Basic example failed: {e}")
    
    try:
        example_with_partition_manager()
    except Exception as e:
        print(f"Partition manager example failed: {e}")
    
    try:
        example_attention_calculator_usage()
    except Exception as e:
        print(f"Calculator example failed: {e}")
    
    try:
        example_hierarchical_processing()
    except Exception as e:
        print(f"Hierarchical example failed: {e}")
    
    try:
        example_performance_comparison()
    except Exception as e:
        print(f"Performance example failed: {e}")
    
    print("\n" + "=" * 50)
    print("Examples complete!")
    
    # Migration guide
    print("\n=== Migration Guide ===")
    print("""
To migrate existing code to use dual-level processing:

1. Replace imports:
   - QwQModel -> QwQModelDualLevel
   - AttentionCalculator -> DualLevelAttentionCalculator

2. Enable dual-level processing:
   model = QwQModelDualLevel(path, enable_dual_level=True)

3. Provide segment information:
   - Create segment dictionaries with id, content, start_pos, end_pos
   - Optionally provide adjacency matrix and edge types

4. Use new methods:
   - extract_dual_level_attention() instead of extract_attention()
   - segment_with_dual_attention() for graph-aware segmentation
   - get_dual_level_embeddings() for both token and node embeddings

5. Process results:
   - Check for 'dual_level' flag in results
   - Access node_attention and cross_attention in layer data
   - Use enhanced calculator for comprehensive analysis
""")


if __name__ == "__main__":
    main()