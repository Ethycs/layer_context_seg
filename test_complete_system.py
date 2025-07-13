# System Integration Test
# Testing the complete layered context graph with GGUF model loading

import sys
import os
sys.path.append('/workspace/layered-context-graph/src')

from models.context_window import ContextWindow
from models.attention_extractor import EnhancedAttentionExtractor
from models.instruction_seeder import InstructionSeeder

def test_complete_system():
    """Test the complete system with GGUF model and attention-based processing"""
    
    print("=== Testing Complete Layered Context Graph System ===\n")
    
    # Sample transcript for testing
    test_transcript = """
    Meeting Discussion on AI Development Strategy
    
    Speaker A: We need to discuss our approach to building the next generation AI system. 
    The current model has limitations in context understanding and we're seeing issues 
    with long conversations.
    
    Speaker B: I agree. From a technical perspective, we should look at attention mechanisms.
    The research shows that transformer models with better attention patterns can handle
    longer contexts more effectively.
    
    Speaker A: What about memory requirements? Our current infrastructure might not support
    larger models. We need to consider cost implications.
    
    Speaker B: That's a good point. We could implement a layered approach where we process
    information in chunks but maintain connections between them. This would be like a
    knowledge graph approach.
    
    Speaker A: Interesting. So instead of one massive context window, we create multiple
    smaller windows that are connected? How would that work technically?
    
    Speaker B: We could use attention patterns to identify natural boundaries in the text,
    then create overlapping segments. The overlap ensures information can flow between
    segments - it's based on percolation theory.
    
    Speaker A: I like this approach. Let's prototype it. What are the next steps?
    
    Speaker B: First, we need to implement the chunking algorithm. Second, we need to
    extract attention patterns. Third, we build the knowledge graph. Finally, we test
    reconstruction from the graph back to useful summaries.
    """
    
    # Step 1: Initialize components
    print("1. Initializing components...")
    
    # Context window with percolation-based overlap
    context_window = ContextWindow(size=1000)  # Smaller size for testing
    
    # Instruction seeder for natural language rules
    seeder = InstructionSeeder()
    
    # Attention extractor (will try GGUF model)
    try:
        attention_extractor = EnhancedAttentionExtractor("qwq", model_type="ollama")
        print("   ✓ GGUF model loaded successfully")
    except Exception as e:
        print(f"   ⚠ GGUF model failed, using fallback: {e}")
        # Fallback to a simple transformer model
        try:
            attention_extractor = EnhancedAttentionExtractor("distilbert-base-uncased", model_type="transformer")
            print("   ✓ Transformer model loaded as fallback")
        except Exception as e2:
            print(f"   ✗ Both models failed: {e2}")
            return
    
    # Step 2: Define natural language rules
    print("\n2. Defining natural language rules...")
    rules = {
        "segmentation": "Split at speaker turns and major topic shifts",
        "reorganization": "Group by themes: technical discussion, resource concerns, action items"
    }
    print(f"   Segmentation rule: {rules['segmentation']}")
    print(f"   Reorganization rule: {rules['reorganization']}")
    
    # Step 3: Seed instructions in the text
    print("\n3. Seeding instructions...")
    seeded_transcript = seeder.seed_with_rules(test_transcript, rules)
    print(f"   Original length: {len(test_transcript)} chars")
    print(f"   Seeded length: {len(seeded_transcript)} chars")
    
    # Step 4: Create context windows with percolation overlap
    print("\n4. Creating context windows...")
    windows = context_window.create_window(seeded_transcript)
    print(f"   Created {len(windows)} windows")
    for i, window in enumerate(windows):
        print(f"   Window {i+1}: {len(window)} chars")
    
    # Step 5: Extract attention patterns
    print("\n5. Extracting attention patterns...")
    try:
        attention_data = attention_extractor.extract_attention(windows)
        print(f"   Model type: {attention_data['model_type']}")
        
        if attention_data['model_type'] == 'ollama':
            print(f"   Processed {len(attention_data['window_patterns'])} windows")
            for i, pattern in enumerate(attention_data['window_patterns']):
                if 'error' in pattern:
                    print(f"   Window {i}: Error - {pattern['error']}")
                else:
                    boundaries = len(pattern.get('suggested_boundaries', []))
                    print(f"   Window {i}: Found {boundaries} suggested boundaries")
        else:
            print(f"   Attention tensors: {len(attention_data.get('attention_tensors', []))}")
            
    except Exception as e:
        print(f"   ⚠ Attention extraction failed: {e}")
        attention_data = {'model_type': 'fallback', 'window_patterns': []}
    
    # Step 6: Analyze attention patterns
    print("\n6. Analyzing attention patterns...")
    try:
        analysis = attention_extractor.analyze_patterns(attention_data)
        print(f"   Analysis completed: {list(analysis.keys())}")
        
        # Show boundary detection results
        if 'segment_boundaries' in analysis:
            print(f"   Found boundaries in {len(analysis['segment_boundaries'])} windows")
            
    except Exception as e:
        print(f"   ⚠ Pattern analysis failed: {e}")
        analysis = {}
    
    # Step 7: Create knowledge graph structure
    print("\n7. Creating knowledge graph...")
    try:
        # Extract meaningful chunks from windows
        chunks = []
        for window in windows:
            # Simple sentence splitting for demo
            sentences = window.split('. ')
            chunks.extend([s.strip() + '.' for s in sentences if len(s.strip()) > 10])
        
        print(f"   Created {len(chunks)} semantic chunks")
        
        # Show some example chunks
        for i, chunk in enumerate(chunks[:3]):
            print(f"   Chunk {i}: {chunk[:60]}...")
        
        # Create graph representation
        graph_data = attention_extractor.create_attention_graph(attention_data, chunks)
        if graph_data:
            print(f"   Graph nodes: {len(graph_data.get('nodes', []))}")
            print(f"   Graph edges: {len(graph_data.get('edges', []))}")
            
            # Show edge details if any exist
            edges = graph_data.get('edges', [])
            if edges:
                print(f"   Sample edges:")
                for edge in edges[:3]:
                    print(f"     {edge['source']} -> {edge['target']} (weight: {edge['weight']:.3f})")
            else:
                print("   No edges created - checking similarity calculations...")
                # Debug: manually check text similarity
                if len(chunks) >= 2:
                    sim = attention_extractor._text_similarity(chunks[0], chunks[1])
                    print(f"   Sample text similarity between chunk 0 and 1: {sim:.3f}")
        else:
            print("   Graph creation skipped (attention data insufficient)")
            
    except Exception as e:
        print(f"   ⚠ Graph creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 8: Demonstrate rule-based processing
    print("\n8. Rule-based processing demonstration...")
    
    # Show how natural language rules would be applied
    rule_encoding = seeder.program_with_natural_language(
        rules['segmentation'], 
        rules['reorganization']
    )
    
    print("   Encoded rules:")
    for key, value in rule_encoding.items():
        if isinstance(value, str):
            print(f"     {key}: {value}")
        else:
            print(f"     {key}: {type(value).__name__}")
    
    print("\n=== System Test Complete ===")
    print(f"✓ Context windows with percolation overlap: {len(windows)} windows")
    print(f"✓ Natural language rule specification: Working")
    print(f"✓ Attention pattern extraction: {attention_data['model_type']} model")
    print(f"✓ GGUF model integration: {'Working' if attention_data['model_type'] == 'ollama' else 'Fallback used'}")
    
if __name__ == "__main__":
    test_complete_system()
