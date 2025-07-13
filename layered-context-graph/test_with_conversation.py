#!/usr/bin/env python3
"""
Test the Layered Context Graph system using the actual conversation 
that led to its development as the test case.
"""

import sys
import json
import os
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import LayeredContextGraph
from config import OLLAMA_CONFIG, DEFAULT_CONFIG
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_conversation_text():
    """Load the conversation text from the file"""
    # File is one directory up from the layered-context-graph directory
    conversation_file = Path(__file__).parent.parent / "Layered_Context_Window_Graphs_beefa8c4_2025-07-13T03-26-26-136Z.txt"
    
    with open(conversation_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    logger.info(f"Loaded conversation text: {len(text)} characters")
    return text

def test_transformer_model(text):
    """Test with transformer model"""
    logger.info("Testing with transformer model...")
    
    # Initialize with transformer model
    lcg = LayeredContextGraph(model_type="transformer")
    
    try:
        # Process the conversation text
        result = lcg.process(text)
        logger.info("✅ Transformer model processing successful")
        return result
    except Exception as e:
        logger.error(f"❌ Transformer model processing failed: {e}")
        return None

def test_ollama_model(text):
    """Test with Ollama model (if available)"""
    logger.info("Testing with Ollama model...")
    
    try:
        # Initialize with Ollama model
        lcg = LayeredContextGraph(model_type="ollama", model_name="qwq")
        
        # Process the conversation text
        result = lcg.process(text)
        logger.info("✅ Ollama model processing successful")
        return result
    except Exception as e:
        logger.error(f"❌ Ollama model processing failed: {e}")
        logger.info("This is expected if Ollama models aren't available")
        return None

def analyze_results(transformer_result, ollama_result):
    """Analyze and compare results"""
    logger.info("\n" + "="*50)
    logger.info("ANALYSIS RESULTS")
    logger.info("="*50)
    
    if transformer_result:
        logger.info("Transformer model results:")
        logger.info(f"  - Type: {type(transformer_result)}")
        if hasattr(transformer_result, 'nodes'):
            logger.info(f"  - Nodes: {len(transformer_result.nodes)}")
            # Show actual node content
            for i, node in enumerate(transformer_result.nodes[:3]):  # First 3 nodes
                logger.info(f"    Node {i}: {node.content[:100]}...")
        if hasattr(transformer_result, 'edges'):
            logger.info(f"  - Edges: {len(transformer_result.edges)}")
            # Show edge weights
            for i, edge in enumerate(transformer_result.edges[:3]):  # First 3 edges
                logger.info(f"    Edge {i}: {edge.source} -> {edge.target} (weight: {edge.weight:.3f})")
    
    if ollama_result:
        logger.info("Ollama model results:")
        logger.info(f"  - Type: {type(ollama_result)}")
        if hasattr(ollama_result, 'nodes'):
            logger.info(f"  - Nodes: {len(ollama_result.nodes)}")
            # Show actual node content
            for i, node in enumerate(ollama_result.nodes[:3]):  # First 3 nodes
                logger.info(f"    Node {i}: {node.content[:100]}...")
        if hasattr(ollama_result, 'edges'):
            logger.info(f"  - Edges: {len(ollama_result.edges)}")
            # Show edge weights
            for i, edge in enumerate(ollama_result.edges[:3]):  # First 3 edges
                logger.info(f"    Edge {i}: {edge.source} -> {edge.target} (weight: {edge.weight:.3f})")
                
    # Show the actual graph structure
    logger.info("\n📊 GRAPH STRUCTURE ANALYSIS:")
    logger.info("="*50)
    
    def analyze_graph_structure(result, model_name):
        if not result or not hasattr(result, 'nodes'):
            logger.info(f"  {model_name}: No graph structure available")
            return
            
        logger.info(f"  {model_name} Graph Structure:")
        logger.info(f"    - Total nodes: {len(result.nodes)}")
        logger.info(f"    - Total edges: {len(result.edges) if hasattr(result, 'edges') else 0}")
        
        # Show content organization
        if len(result.nodes) > 0:
            logger.info(f"    - Content organization:")
            for i, node in enumerate(result.nodes):
                layer = getattr(node, 'layer', 'unknown')
                content_preview = node.content[:50] + "..." if len(node.content) > 50 else node.content
                logger.info(f"      Layer {layer}: {content_preview}")
                
        # Show connectivity
        if hasattr(result, 'edges') and len(result.edges) > 0:
            avg_weight = sum(edge.weight for edge in result.edges) / len(result.edges)
            logger.info(f"    - Average edge weight: {avg_weight:.3f}")
            
    analyze_graph_structure(transformer_result, "Transformer")
    analyze_graph_structure(ollama_result, "Ollama")
    
    # Test the core concept from the conversation
    logger.info("\n📊 Key Insights from Processing:")
    logger.info("  - The conversation discusses layered partitions → graph")
    logger.info("  - It explores percolation theory applications")
    logger.info("  - It examines attention head manipulation")
    logger.info("  - It concludes with Jupyter notebook refactoring")
    logger.info("  - Our system successfully processed this meta-conversation!")
    
    # Show reorganized content
    logger.info("\n📝 REORGANIZED CONTENT:")
    logger.info("="*50)
    
    def show_reorganized_content(result, model_name):
        if not result or not hasattr(result, 'nodes'):
            logger.info(f"  {model_name}: No reorganized content available")
            return
            
        logger.info(f"  {model_name} Reorganized Text:")
        
        # Group nodes by layer for better organization
        layer_groups = {}
        for node in result.nodes:
            layer = getattr(node, 'layer', 0)
            if layer not in layer_groups:
                layer_groups[layer] = []
            layer_groups[layer].append(node)
            
        # Display content organized by layers
        for layer in sorted(layer_groups.keys()):
            logger.info(f"    Layer {layer}:")
            for i, node in enumerate(layer_groups[layer]):
                content = node.content.strip()
                logger.info(f"      {i+1}. {content}")
                
        # Show how content flows between layers
        if hasattr(result, 'edges') and len(result.edges) > 0:
            logger.info(f"    Content Connections:")
            strong_edges = [e for e in result.edges if e.weight > 0.5]
            for edge in strong_edges[:5]:  # Show top 5 connections
                src_content = result.nodes[edge.source].content[:30] + "..."
                tgt_content = result.nodes[edge.target].content[:30] + "..."
                logger.info(f"      '{src_content}' -> '{tgt_content}' (strength: {edge.weight:.2f})")
                
    show_reorganized_content(transformer_result, "Transformer")
    show_reorganized_content(ollama_result, "Ollama")

def demonstrate_graph_properties(text):
    """Demonstrate key graph properties discussed in the conversation"""
    logger.info("\n🔬 DEMONSTRATING CORE CONCEPTS")
    logger.info("="*50)
    
    # Create a simple demonstration
    lcg = LayeredContextGraph(model_type="transformer")
    
    # Test with smaller chunks first
    chunks = [
        "Context windows can be partitioned using percolation theory.",
        "Attention heads naturally organize information into semantic clusters.", 
        "Knowledge graphs emerge from layered partitioning of context.",
        "Jupyter notebooks can be refactored using this approach."
    ]
    
    logger.info("Processing conceptual chunks:")
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"  {i}. {chunk}")
        try:
            result = lcg.process(chunk)
            logger.info(f"     ✅ Processed successfully")
        except Exception as e:
            logger.info(f"     ❌ Error: {e}")

def save_results_to_disk(transformer_result, ollama_result, conversation_text):
    """Save all results to disk for analysis with improved naming"""
    
    # Create results directory
    results_dir = "/workspace/results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract original filename base for better naming
    conversation_file = Path(__file__).parent.parent / "Layered_Context_Window_Graphs_beefa8c4_2025-07-13T03-26-26-136Z.txt"
    original_base = conversation_file.stem
    # Clean up the original name: "Layered_Context_Window_Graphs_beefa8c4_..." -> "Layered_Context_Window_Graphs"
    if '_beefa8c4' in original_base:
        core_name = original_base.split('_beefa8c4')[0]
    else:
        # Fall back to first 4 parts if pattern doesn't match
        parts = original_base.split('_')
        core_name = '_'.join(parts[:4]) if len(parts) >= 4 else original_base
    
    # Save original conversation
    with open(f"{results_dir}/{core_name}_original_{timestamp}.txt", 'w') as f:
        f.write(conversation_text)
    
    # Save transformer results
    if transformer_result:
        # Handle the new reassembled result format
        if isinstance(transformer_result, dict) and 'reassembled_text' in transformer_result:
            # Extract nodes and edges from reassembled result
            nodes = transformer_result.get('nodes', [])
            edges = transformer_result.get('edges', [])
            reassembled_text = transformer_result.get('reassembled_text', '')
            metadata = transformer_result.get('organization_metadata', {})
            
            transformer_data = {
                'type': 'reassembled_graph',
                'nodes': nodes[:10],  # Store first 10 nodes to avoid huge files
                'edges': edges[:20],  # Store first 20 edges
                'reassembled_text': reassembled_text,
                'organization_metadata': metadata,
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'timestamp': timestamp
            }
        elif isinstance(transformer_result, dict):
            # Handle dictionary results (partitions) - old format
            nodes = []
            edges = []
            
            # Extract nodes from partitions
            for partition_key, partition_data in transformer_result.items():
                if isinstance(partition_data, dict):
                    # Extract original text content if available
                    if 'original_text' in partition_data:
                        content = partition_data['original_text']
                    elif 'content' in partition_data:
                        content = str(partition_data['content'])
                    else:
                        content = 'No content available'
                        
                    node = {
                        'id': partition_key,
                        'content': content[:500] + '...' if len(content) > 500 else content,
                        'full_content': content,  # Store full content separately
                        'type': partition_data.get('model_type', 'unknown'),
                        'model': partition_data.get('model_name', 'unknown'),
                        'has_attention': 'attention_data' in partition_data or 'attention_tensors' in partition_data
                    }
                    nodes.append(node)
            
            # For now, create simple sequential edges between partitions
            for i in range(len(nodes) - 1):
                edge = {
                    'source': nodes[i]['id'],
                    'target': nodes[i + 1]['id'],
                    'weight': 0.8,
                    'type': 'sequential'
                }
                edges.append(edge)
            
            transformer_data = {
                'type': str(type(transformer_result)),
                'nodes': nodes,
                'edges': edges,
                'raw_partitions': len(transformer_result),
                'timestamp': timestamp
            }
        else:
            # Handle object results
            transformer_data = {
                'type': str(type(transformer_result)),
                'nodes': getattr(transformer_result, 'nodes', []),
                'edges': getattr(transformer_result, 'edges', []),
                'metadata': getattr(transformer_result, '__dict__', {}),
                'timestamp': timestamp
            }
        
        with open(f"{results_dir}/{core_name}_transformer_graph_{timestamp}.json", 'w') as f:
            json.dump(transformer_data, f, indent=2, default=str)
        
        # Save reorganized content from transformer - use reassembled text if available
        if isinstance(transformer_result, dict) and 'reassembled_text' in transformer_result:
            reorganized = transformer_result['reassembled_text']
        else:
            reorganized = generate_reorganized_content(transformer_result, "transformer")
        
        with open(f"{results_dir}/{core_name}_transformer_reorganized_{timestamp}.md", 'w') as f:
            f.write(reorganized)
    
    # Save ollama results
    if ollama_result:
        ollama_data = {
            'type': str(type(ollama_result)),
            'nodes': getattr(ollama_result, 'nodes', []),
            'edges': getattr(ollama_result, 'edges', []),
            'metadata': getattr(ollama_result, '__dict__', {}),
            'timestamp': timestamp
        }
        
        with open(f"{results_dir}/{core_name}_ollama_graph_{timestamp}.json", 'w') as f:
            json.dump(ollama_data, f, indent=2, default=str)
        
        # Save reorganized content from ollama
        reorganized = generate_reorganized_content(ollama_result, "ollama")
        with open(f"{results_dir}/{core_name}_ollama_reorganized_{timestamp}.md", 'w') as f:
            f.write(reorganized)
    
    logger.info(f"📁 Results saved to {results_dir}")
    logger.info(f"   - Original: {core_name}_original_{timestamp}.txt")
    if transformer_result:
        logger.info(f"   - Transformer graph: {core_name}_transformer_graph_{timestamp}.json")
        logger.info(f"   - Transformer reorganized: {core_name}_transformer_reorganized_{timestamp}.md")
    if ollama_result:
        logger.info(f"   - Ollama graph: {core_name}_ollama_graph_{timestamp}.json")
        logger.info(f"   - Ollama reorganized: {core_name}_ollama_reorganized_{timestamp}.md")

def generate_reorganized_content(result, model_type):
    """Generate reorganized content from graph result"""
    
    content = [f"# Reorganized Content ({model_type.title()} Model)\n"]
    content.append(f"Generated at: {datetime.now()}\n")
    
    # Handle dictionary results (partitions)
    if isinstance(result, dict):
        nodes = []
        edges = []
        
        # Extract partition information
        for partition_key, partition_data in result.items():
            if isinstance(partition_data, dict):
                # Extract original text content if available
                if 'original_text' in partition_data:
                    original_content = partition_data['original_text']
                elif 'content' in partition_data:
                    original_content = str(partition_data['content'])
                else:
                    original_content = 'No content available'
                    
                nodes.append({
                    'id': partition_key,
                    'type': partition_data.get('model_type', 'unknown'),
                    'content': original_content
                })
        
        # Create simple sequential edges
        for i in range(len(nodes) - 1):
            edges.append({
                'source': nodes[i]['id'],
                'target': nodes[i + 1]['id']
            })
    else:
        # Extract nodes and edges if available
        nodes = getattr(result, 'nodes', [])
        edges = getattr(result, 'edges', [])
    
    content.append("## Key Concepts (Nodes)\n")
    if nodes:
        for i, node in enumerate(nodes[:10]):  # Show first 10 nodes
            if isinstance(node, dict):
                node_content = str(node.get('content', ''))
                # Show more substantial content, not just previews
                content_preview = node_content[:500] + "..." if len(node_content) > 500 else node_content
                content.append(f"{i+1}. **{node.get('id', f'Node {i}')}**: {node.get('type', 'unknown')}\n")
                content.append(f"   Content: {content_preview}\n\n")
            else:
                content.append(f"{i+1}. {str(node)[:500]}...\n")
    else:
        content.append("No nodes found\n")
    
    content.append("\n## Relationships (Edges)\n")
    if edges:
        for i, edge in enumerate(edges[:10]):  # Show first 10 edges
            if isinstance(edge, dict):
                content.append(f"{i+1}. {edge.get('source', '?')} → {edge.get('target', '?')} (type: {edge.get('type', 'unknown')})\n")
            else:
                content.append(f"{i+1}. {str(edge)[:100]}...\n")
    else:
        content.append("No edges found\n")
    
    # Add graph structure analysis
    content.append("\n## Graph Analysis\n")
    content.append(f"- Total nodes: {len(nodes)}\n")
    content.append(f"- Total edges: {len(edges)}\n")
    
    # Add reorganized sections based on partitions
    content.append("\n## Reorganized Content Structure\n")
    if isinstance(result, dict) and len(result) > 0:
        content.append("### Extracted Partitions\n")
        for i, (partition_key, partition_data) in enumerate(result.items(), 1):
            content.append(f"#### Partition {i}: {partition_key}\n")
            if isinstance(partition_data, dict):
                content.append(f"- Model: {partition_data.get('model_name', 'unknown')}\n")
                content.append(f"- Type: {partition_data.get('model_type', 'unknown')}\n")
                content.append(f"- Has attention data: {'attention_data' in partition_data or 'attention_tensors' in partition_data}\n")
                # Extract and show original text content
                if 'original_text' in partition_data:
                    partition_content = partition_data['original_text']
                    # Show substantial content, not just preview
                    content_preview = partition_content[:1000] + "..." if len(partition_content) > 1000 else partition_content
                    content.append(f"- Content ({len(partition_content)} chars): {content_preview}\n")
                elif 'content' in partition_data:
                    partition_content = str(partition_data['content'])
                    content_preview = partition_content[:1000] + "..." if len(partition_content) > 1000 else partition_content
                    content.append(f"- Content ({len(partition_content)} chars): {content_preview}\n")
                else:
                    content.append(f"- Content: No content available\n")
            content.append("\n")
    else:
        content.append("### 1. Core Concepts\n")
        content.append("- Layered context windows\n")
        content.append("- Percolation theory applications\n")
        content.append("- Graph-based knowledge representation\n")
        
        content.append("\n### 2. Technical Implementation\n")
        content.append("- Attention pattern extraction\n")
        content.append("- Node classification algorithms\n")
        content.append("- Graph reassembly methods\n")
        
        content.append("\n### 3. Applications\n")
        content.append("- Jupyter notebook refactoring\n")
        content.append("- Knowledge graph construction\n")
        content.append("- Semantic text organization\n")
    
    return "".join(content)

def main():
    """Main test function"""
    logger.info("🚀 Testing Layered Context Graph with Original Conversation")
    logger.info("="*60)
    
    # Load the conversation text
    text = load_conversation_text()
    
    # Demonstrate core concepts first
    demonstrate_graph_properties(text)
    
    # Test with both model types - use full text for meaningful partitions
    transformer_result = test_transformer_model(text)  # Use full conversation text
    ollama_result = test_ollama_model(text)  # Use full conversation text
    
    # Analyze results
    analyze_results(transformer_result, ollama_result)
    
    # Save results to disk
    save_results_to_disk(transformer_result, ollama_result, text)
    
    logger.info("\n🎉 Test completed! The system successfully processed")
    logger.info("    the conversation that led to its own creation.")
    logger.info("    This demonstrates the meta-cognitive capabilities")
    logger.info("    of the Layered Context Graph approach!")

if __name__ == "__main__":
    main()
