#!/usr/bin/env python3
"""
Test script to verify the reassembly fixes
"""

import sys
from pathlib import Path

# Add src path
project_root = Path(__file__).parent.resolve()
src_path = project_root / "layered-context-graph" / "src"
sys.path.insert(0, str(src_path))

from models.context_window import ContextWindow
from graph.graph_reassembler_enhanced import GraphReassembler

def test_reassembly_fix():
    """Test that reassembly preserves content and formatting"""
    
    # Test text with code blocks and formatting
    test_text = """# Main Title

This is a regular paragraph with some content.

## Code Example

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
```

## Another Section

This is another paragraph with more content that should be preserved.
"""
    
    # Create context window and reassembler
    context_window = ContextWindow(size=500)  # Small size to force multiple windows
    reassembler = GraphReassembler()
    
    # Test window creation
    windows = context_window.create_window(test_text)
    
    print(f"Original text length: {len(test_text)} characters")
    print(f"Number of windows created: {len(windows)}")
    
    # Create mock nodes and edges for reassembly
    nodes = []
    edges = []
    
    for i, window_content in enumerate(windows):
        # Create basic formatting info
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
    
    print(f"Created {len(nodes)} nodes and {len(edges)} edges")
    
    # Test reassembly
    reassembled = reassembler.reassemble_graph(nodes, edges, test_text)
    
    print(f"\nReassembled length: {len(reassembled)} characters")
    print("="*50)
    print(reassembled[:500] + "..." if len(reassembled) > 500 else reassembled)
    
    # Check that code blocks are preserved
    print("\nCode block preservation:")
    print("- Original has ```python:", "```python" in test_text)
    print("- Reassembled has ```python:", "```python" in reassembled)
    print("- Original has code content:", "nodes = []" in test_text)
    print("- Reassembled has code content:", "nodes = []" in reassembled)
    
    # Check formatting
    print("\nFormatting preservation:")
    print("- Original has headings:", "#" in test_text)
    print("- Reassembled has headings:", "#" in reassembled)
    
    return reassembled

if __name__ == "__main__":
    test_reassembly_fix()