#!/usr/bin/env python3
"""
Test Script for Attention-Based Natural Language Guided Processor
---------------------------------------------------------------
This script demonstrates the enhanced attention-based natural language 
guided processor using both transformer and Ollama models.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import the language guided processor
from processor.language_guided_processor import LanguageGuidedProcessor

# Define the results directory
RESULTS_DIR = Path("/workspace/results")
RESULTS_DIR.mkdir(exist_ok=True)

def test_with_text(text, rules=None, model_type="transformer", model_source="distilbert-base-uncased"):
    """
    Test the language guided processor with the given text and rules
    
    Args:
        text: Text to process
        rules: Dictionary with segmentation and reorganization rules
        model_type: 'transformer' or 'ollama'
        model_source: Model name or path
    """
    print(f"\n\n{'='*50}")
    print(f"Testing with {model_type} model: {model_source}")
    print(f"{'='*50}")
    
    if rules:
        print(f"Segmentation rule: {rules.get('segmentation', 'Default')}")
        print(f"Reorganization rule: {rules.get('reorganization', 'Default')}")
    
    # Create the processor
    processor = LanguageGuidedProcessor(
        model_source=model_source,
        model_type=model_type
    )
    
    # Process the text
    start_time = time.time()
    result = processor.process_document(text, rules)
    processing_time = time.time() - start_time
    
    # Print basic stats
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    print(f"Created {len(result['segments'])} segments")
    
    if result.get('graph'):
        print(f"Graph has {len(result['graph']['nodes'])} nodes and {len(result['graph']['edges'])} edges")
    
    # Save the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a descriptive filename
    model_name = model_source.split('/')[-1]
    output_name = f"NL_Guided_Processor_{model_type}_{timestamp}"
    
    # Save the reorganized output
    if result.get('outputs') and 'summary' in result['outputs']:
        output_file = RESULTS_DIR / f"{output_name}_summary.md"
        with open(output_file, "w") as f:
            f.write(f"# Summary from {model_type.capitalize()} Model\n\n")
            f.write(f"Model: {model_source}\n\n")
            if rules:
                f.write(f"Segmentation rule: {rules.get('segmentation', 'Default')}\n")
                f.write(f"Reorganization rule: {rules.get('reorganization', 'Default')}\n\n")
            f.write(result['outputs']['summary'])
        
        print(f"\nSummary saved to {output_file}")
    
    # Save all outputs
    all_outputs_file = RESULTS_DIR / f"{output_name}_all_outputs.md"
    with open(all_outputs_file, "w") as f:
        f.write(f"# All Outputs from {model_type.capitalize()} Model\n\n")
        f.write(f"Model: {model_source}\n\n")
        if rules:
            f.write(f"Segmentation rule: {rules.get('segmentation', 'Default')}\n")
            f.write(f"Reorganization rule: {rules.get('reorganization', 'Default')}\n\n")
        
        for output_type, content in result['outputs'].items():
            f.write(f"\n## {output_type.capitalize()}\n\n")
            f.write(content)
    
    print(f"All outputs saved to {all_outputs_file}")
    
    # Save the graph as JSON for visualization
    graph_file = RESULTS_DIR / f"{output_name}_graph.json"
    with open(graph_file, "w") as f:
        # Replace node content with shorter versions for visualization
        if result.get('graph') and 'nodes' in result['graph']:
            for node in result['graph']['nodes']:
                if 'content' in node and len(node['content']) > 100:
                    node['full_content'] = node['content']
                    node['content'] = node['content'][:100] + "..."
        
        json.dump(result.get('graph', {}), f, indent=2)
    
    print(f"Graph saved to {graph_file}")
    
    return result

def load_test_data(file_path):
    """Load test data from a file"""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading test data: {e}")
        return "This is a fallback test text since the real test data could not be loaded."

def main():
    """Main test function"""
    # Define some natural language rules for testing
    rules_sets = [
        {
            'segmentation': "Split at major topic changes and whenever the speaker changes",
            'reorganization': "Group by theme, with most important information first"
        },
        {
            'segmentation': "Create new segments for each question and its answer",
            'reorganization': "Order chronologically, maintaining the flow of conversation"
        },
        {
            'segmentation': "Split whenever there's a transition between concepts or ideas",
            'reorganization': "Group similar concepts together for better understanding"
        }
    ]
    
    # Test with a sample text from the condensed architecture document
    arch_path = Path(__file__).parent / "condensed_arch.md"
    if arch_path.exists():
        arch_text = load_test_data(arch_path)
        print("\nTesting with condensed architecture document...")
        
        # Test with transformer model
        test_with_text(
            arch_text, 
            rules=rules_sets[0], 
            model_type="transformer",
            model_source="distilbert-base-uncased"
        )
        
        # Test with Ollama model if available
        if Path("/workspace/qwq.gguf").exists():
            test_with_text(
                arch_text, 
                rules=rules_sets[0], 
                model_type="ollama",
                model_source="qwq"
            )
    else:
        print(f"Test file not found: {arch_path}")
    
    # Test with a conversation example
    conversation_text = """
    User: What's the best way to implement attention mechanisms in transformer models?
    
    Claude: Implementing attention mechanisms in transformer models involves several key steps:
    
    1. **Query, Key, Value Projections**: First, project your input embeddings into query (Q), key (K), and value (V) spaces using separate linear transformations.
    
    2. **Attention Scores**: Compute attention scores by taking the dot product of queries with keys, scaled by √d_k where d_k is the dimension of the key vectors.
    
    3. **Softmax**: Apply softmax to get attention weights that sum to 1.
    
    4. **Weighted Sum**: Multiply these weights by the value vectors and sum them to get the attention output.
    
    5. **Multi-Head Attention**: For multi-head attention, perform these operations in parallel with different projection matrices, then concatenate and project the results.
    
    Here's a simplified implementation in PyTorch:
    
    [CODE EXAMPLE: MultiHeadAttention class implementation with forward method]
    
    The beauty of attention is that it allows the model to focus on different parts of the input sequence dynamically, which is crucial for understanding context and relationships between tokens.
    
    User: How can I visualize attention patterns to understand what my model is focusing on?
    
    Claude: Visualizing attention patterns is extremely helpful for understanding and debugging transformer models. Here are several effective approaches:

    1. **Attention Heatmaps**: The most common visualization is a heatmap showing which tokens attend to which other tokens.
    
    [CODE EXAMPLE: plot_attention function implementation]

    2. **Interactive Visualizations**: For more dynamic exploration, use tools like BertViz or the Transformers Interpretability Library.
    
    [CODE EXAMPLE: Using bertviz for head_view visualization]

    3. **Attention Flow Graphs**: Visualize how attention flows through the network.
    
    [CODE EXAMPLE: plot_attention_flow function using networkx]

    4. **Attention by Layer and Head**: Create a grid of heatmaps to compare different heads.
    
    [CODE EXAMPLE: plot_attention_heads function implementation]

    5. **Attention on Text**: For NLP applications, highlight text based on attention.
    
    [CODE EXAMPLE: highlight_text function using HTML]

    These visualizations can reveal fascinating patterns:
    - Some heads focus on adjacent tokens (local attention)
    - Others connect related words across long distances
    - Some track syntactic structure like subject-verb relationships
    - Later layers often capture more abstract relationships

    Would you like me to explain how to interpret specific patterns you might see in these visualizations?
    """
    
    print("\nTesting with conversation text...")
    test_with_text(
        conversation_text, 
        rules=rules_sets[1], 
        model_type="transformer",
        model_source="distilbert-base-uncased"
    )

if __name__ == "__main__":
    main()
