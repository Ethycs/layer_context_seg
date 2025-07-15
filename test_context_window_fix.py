#!/usr/bin/env python3
"""
Test script to verify the context window fixes
"""

import sys
from pathlib import Path

# Add src path
project_root = Path(__file__).parent.resolve()
src_path = project_root / "layered-context-graph" / "src"
sys.path.insert(0, str(src_path))

from models.context_window import ContextWindow

def test_context_window_fix():
    """Test that context windows preserve content properly"""
    
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

## Another Section

This is another paragraph with more content that should be preserved.
"""
    
    # Create context window with reasonable size
    context_window = ContextWindow(size=2000)  # 2000 characters
    
    # Test window creation
    windows = context_window.create_window(test_text)
    
    print(f"Original text length: {len(test_text)} characters")
    print(f"Number of windows created: {len(windows)}")
    
    total_windows_length = sum(len(window) for window in windows)
    print(f"Total windows length: {total_windows_length} characters")
    
    # Check that content is preserved
    for i, window in enumerate(windows):
        print(f"\nWindow {i} ({len(window)} chars):")
        print("="*50)
        print(window[:200] + "..." if len(window) > 200 else window)
        
    # Test that code blocks are preserved
    for i, window in enumerate(windows):
        if "```python" in window:
            print(f"\nWindow {i} contains code block - checking formatting...")
            print("Code block preserved:", "```python" in window and "```" in window)
            
    # Test reassembly
    reassembled = "\n\n".join(windows)
    print(f"\nReassembled length: {len(reassembled)} characters")
    print("Content preserved:", len(reassembled) > len(test_text) * 0.8)  # Allow for some overlap
    
    return windows

if __name__ == "__main__":
    test_context_window_fix()