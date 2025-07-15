#!/usr/bin/env python3
"""
Test the master processor to ensure it works with the fixes
"""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.resolve()
src_path = project_root / "layered-context-graph" / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

def test_master_processor():
    """Test the master processor with a simple example"""
    
    # Test content similar to what was causing truncation
    test_text = """# Simple Test Document

This is a test document to verify the master processor.

## Code Example

```python
# Simple function example
def process_data(data):
    results = []
    for item in data:
        processed = {
            'id': item.id,
            'value': item.value,
            'status': 'processed'
        }
        results.append(processed)
    return results
```

## Additional Section

This section contains more text to test the processing pipeline.
The system should preserve all content during processing.
"""
    
    try:
        # Import required modules
        from master_config import get_config
        from master_processor import FullMasterProcessor
        
        # Get configuration
        config = get_config(mode='single-pass', model_type='ollama')
        
        # Create processor
        processor = FullMasterProcessor(config)
        
        print("Testing master processor...")
        print(f"Input text length: {len(test_text)} characters")
        
        # Process the text
        results = processor.process_text(test_text)
        
        print(f"Processing mode: {results['mode']}")
        print(f"Number of nodes: {results['nodes']}")
        print(f"Number of edges: {results['edges']}")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        
        # Check output
        if 'output' in results:
            output = results['output']
            if isinstance(output, str):
                print(f"Output length: {len(output)} characters")
                print(f"Contains code: {'```python' in output}")
                print(f"Contains return: {'return results' in output}")
            elif isinstance(output, dict):
                if 'reassembled_text' in output:
                    reassembled = output['reassembled_text']
                    print(f"Reassembled length: {len(reassembled)} characters")
                    print(f"Contains code: {'```python' in reassembled}")
                    print(f"Contains return: {'return results' in reassembled}")
                    
                    # Show sample
                    print("\nSample output:")
                    print("-" * 40)
                    print(reassembled[:300] + "..." if len(reassembled) > 300 else reassembled)
                
                if 'nodes' in output:
                    nodes = output['nodes']
                    print(f"\nNodes created: {len(nodes)}")
                    total_node_content = sum(len(node.get('content', '')) for node in nodes)
                    print(f"Total node content: {total_node_content} characters")
        
        print("\n✅ Master processor test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing master processor: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_master_processor()