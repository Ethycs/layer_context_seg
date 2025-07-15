#!/usr/bin/env python3
"""
Test the full pipeline to ensure content is not truncated
"""

import sys
from pathlib import Path

# Add src path
project_root = Path(__file__).parent.resolve()
src_path = project_root / "layered-context-graph" / "src"
sys.path.insert(0, str(src_path))

from models.context_window import ContextWindow
from models.percolation_context_window import PercolationContextWindow
from graph.graph_reassembler_enhanced import GraphReassembler
from graph.hierarchical_graph_builder import HierarchicalGraphBuilder

def test_full_pipeline():
    """Test the full pipeline with the problematic content from the user"""
    
    # Use the exact type of content that was causing truncation
    test_text = """# Graph Builder Implementation

This document describes the implementation of a graph builder system.

## Core Architecture

The system consists of several key components:

### Clustering Algorithm

```python
# Each cluster becomes a node
# Attention patterns between clusters become edges
nodes = []
edges = []

for cluster in clusters:
    # Extract entity/concept from cluster
    node = {
        'id': hash(cluster.text),
        'text': cluster.text,
        'type': cluster.instruction_type,  # From seeded instructions
        'embedding': cluster.mean_embedding
    }
    nodes.append(node)

# Inter-cluster attention defines relationships
for i, j in combinations(range(len(clusters)), 2):
    attention_strength = self.cross_cluster_attention(i, j)
    if attention_strength > threshold:
        edges.append({
            'source': nodes[i]['id'],
            'target': nodes[j]['id'],
            'weight': attention_strength,
            'type': self.infer_relation_type(clusters[i], clusters[j])
        })

return nodes, edges
```

### Graph Processing

The graph processing involves several steps:

1. **Node Creation**: Each cluster becomes a node in the graph
2. **Edge Detection**: Attention patterns define relationships
3. **Hierarchical Organization**: Build tree structure from flat graph
4. **Reassembly**: Reconstruct document from graph

## Implementation Details

The implementation uses percolation theory to optimize the overlap between context windows.

### Key Features

- Attention-based edge detection
- Hierarchical graph structure
- Content preservation during reassembly
- Formatting preservation

## Testing

This system has been tested with various input formats including:
- Code blocks
- Markdown formatting
- Complex nested structures
- Long documents with multiple sections

The system maintains content integrity throughout the processing pipeline.
"""
    
    print(f"Original text length: {len(test_text)} characters")
    print(f"Contains code blocks: {'```python' in test_text}")
    print(f"Contains headings: {'#' in test_text}")
    
    # Test with different window sizes
    for window_size in [1000, 2000, 4000]:
        print(f"\n{'='*60}")
        print(f"Testing with window size: {window_size}")
        print(f"{'='*60}")
        
        # Create percolation context window
        percolation_window = PercolationContextWindow(size=window_size, overlap_ratio=0.2)
        
        # Create windows
        windows = percolation_window.create_window(test_text)
        
        print(f"Number of windows: {len(windows)}")
        
        # Show window content lengths
        for i, window in enumerate(windows):
            print(f"Window {i}: {len(window)} chars")
        
        # Create nodes from windows
        nodes = []
        edges = []
        
        for i, window_content in enumerate(windows):
            # Extract formatting info
            formatting = {
                'has_code': '```' in window_content,
                'has_list': any(line.strip().startswith(('-', '*', '+')) for line in window_content.split('\n')),
                'heading_level': None,
                'language': 'python' if '```python' in window_content else None
            }
            
            # Check for headings
            for line in window_content.split('\n'):
                if line.strip().startswith('#'):
                    formatting['heading_level'] = len(line.strip()) - len(line.strip().lstrip('#'))
                    break
            
            node = {
                'id': f'node_{i}',
                'content': window_content,
                'formatting': formatting,
                'importance': 0.8,
                'segment_type': 'code' if formatting['has_code'] else 'text'
            }
            nodes.append(node)
            
            # Create sequential edges
            if i > 0:
                edges.append({
                    'source': f'node_{i-1}',
                    'target': f'node_{i}',
                    'weight': 0.9,
                    'type': 'sequential'
                })
        
        # Test hierarchical processing
        hierarchical_builder = HierarchicalGraphBuilder()
        hierarchical_nodes, tree_edges = hierarchical_builder.build_hierarchy(nodes, edges)
        
        print(f"Hierarchical nodes: {len(hierarchical_nodes)}")
        print(f"Tree edges: {len(tree_edges)}")
        
        # Test reassembly
        reassembler = GraphReassembler()
        reassembled = reassembler.reassemble_graph(hierarchical_nodes, tree_edges, test_text)
        
        print(f"Reassembled length: {len(reassembled)} characters")
        print(f"Content preservation: {len(reassembled) / len(test_text) * 100:.1f}%")
        
        # Check specific content preservation
        print("Content checks:")
        print(f"- Code blocks: {'```python' in reassembled}")
        print(f"- Core function: {'nodes.append(node)' in reassembled}")
        print(f"- Complete edges code: {'edges.append({' in reassembled}")
        print(f"- Return statement: {'return nodes, edges' in reassembled}")
        
        # Show a sample of the reassembled content
        print("\nSample reassembled content:")
        print("-" * 40)
        print(reassembled[:400] + "..." if len(reassembled) > 400 else reassembled)
        
        # Check for truncation issues
        if len(reassembled) < len(test_text) * 0.8:
            print("⚠️  WARNING: Significant content loss detected!")
        elif "return nodes, edges" not in reassembled:
            print("⚠️  WARNING: Code completion missing!")
        else:
            print("✅ Content appears to be well preserved")

if __name__ == "__main__":
    test_full_pipeline()