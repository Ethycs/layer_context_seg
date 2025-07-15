#!/usr/bin/env python3
"""
Test formatting preservation
"""

from master_processor import FullMasterProcessor
from master_config import get_config
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.resolve()
src_path = project_root / "layered-context-graph" / "src"
sys.path.insert(0, str(src_path))

from config.graph_config import get_config as get_graph_config

# Test document with complex formatting
test_document = """# Technical Documentation

## Introduction

This document contains various formatting that should be preserved:

### Code Example

```python
def hello_world():
    # This indentation should be preserved
    print("Hello, World!")
    
    # Multi-line code
    for i in range(5):
        print(f"  Number: {i}")
```

### Conversation Section

Speaker A: Let's discuss the implementation.
Speaker B: I think we should preserve the formatting carefully.
Speaker A: Agreed. The indentation and line breaks are important.

### List Section

Key features:
- Preserves **bold** text
- Maintains *italic* formatting
- Keeps `inline code` intact
- Respects list structure
  - Including nested items
  - With proper indentation

### Indented Code Block

    This is an indented code block
    It should maintain its 4-space indentation
    Across multiple lines

## Conclusion

The formatting preservation system ensures that:

1. Code blocks remain intact
2. Conversations maintain speaker labels
3. Lists preserve their structure
4. Markdown formatting is retained
"""

def test_formatting_preservation():
    print("Testing Formatting Preservation")
    print("=" * 60)
    
    # Save test document
    test_file = Path("test_formatting_doc.md")
    test_file.write_text(test_document)
    
    # Get configurations
    config = get_config(mode='single-pass')
    graph_config = get_graph_config('default')
    
    # Enable all formatting preservation
    graph_config.disassembly.preserve_whitespace = True
    graph_config.disassembly.preserve_code_blocks = True
    graph_config.disassembly.preserve_markdown = True
    graph_config.disassembly.preserve_line_breaks = True
    graph_config.reconstruction.preserve_original_formatting = True
    graph_config.reconstruction.preserve_indentation = True
    graph_config.reconstruction.preserve_empty_lines = True
    graph_config.reconstruction.maintain_code_block_integrity = True
    
    # Process document
    processor = FullMasterProcessor(config, graph_config)
    results = processor.process_text(test_document)
    
    # Get reconstructed document
    if 'output' in results and 'tape2' in results['output']:
        reconstructed = results['output']['tape2']
        
        # Save reconstructed document
        output_file = Path("test_formatting_reconstructed.md")
        output_file.write_text(reconstructed)
        
        print(f"✅ Document processed successfully!")
        print(f"  Nodes created: {results['nodes']}")
        print(f"  Processing time: {results['processing_time']:.2f}s")
        print(f"\nOriginal document saved to: {test_file}")
        print(f"Reconstructed document saved to: {output_file}")
        
        # Check specific formatting elements
        print("\nFormatting Checks:")
        print(f"  - Code blocks preserved: {'```python' in reconstructed}")
        print(f"  - Speaker labels preserved: {'Speaker A:' in reconstructed}")
        print(f"  - List structure preserved: {'- Preserves' in reconstructed}")
        print(f"  - Indented code preserved: {'    This is an indented' in reconstructed}")
        print(f"  - Markdown formatting: {'**bold**' in reconstructed}")
        
    else:
        print("❌ Failed to get reconstructed document")
    
    # Clean up
    test_file.unlink(missing_ok=True)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_formatting_preservation()