# Seed-Focused Condensation Demonstration
# This script shows how to condense a document around its original seed concept

import sys
import os
sys.path.append('/workspace/layered-context-graph/src')

from models.context_window import ContextWindow
from models.attention_extractor import EnhancedAttentionExtractor
from models.instruction_seeder import InstructionSeeder

def condense_around_seed():
    """
    Demonstrate condensing the conversation around the original seed concept:
    'layered partitions of n context windows that interpolated into a graph'
    """
    
    print("=== Seed-Focused Condensation Demonstration ===\n")
    
    # Read the original conversation
    with open('/workspace/layered-context-graph/Layered_Context_Window_Graphs_beefa8c4_2025-07-13T03-26-26-136Z.txt', 'r') as f:
        conversation_text = f.read()
    
    # Identify the seed concept
    seed_concept = "layered partitions of n context windows that interpolated into a graph"
    print(f"🌱 Seed Concept: {seed_concept}")
    print(f"📄 Original document: {len(conversation_text)} characters\n")
    
    # Define seed-focused rules
    seed_rules = {
        "segmentation": f"Split content based on relevance to '{seed_concept}' - keep theoretical foundations, implementations, and mathematical proofs",
        "reorganization": f"Organize around the seed concept '{seed_concept}' - put core theory first, then mathematical foundations, then implementations"
    }
    
    print("🎯 Seed-Focused Rules:")
    print(f"   Segmentation: {seed_rules['segmentation']}")
    print(f"   Reorganization: {seed_rules['reorganization']}\n")
    
    # Initialize components for seed-focused processing
    context_window = ContextWindow(size=2000)
    seeder = InstructionSeeder()
    
    try:
        attention_extractor = EnhancedAttentionExtractor("qwq", model_type="ollama")
        print("   ✓ GGUF model loaded for seed-focused analysis")
    except Exception as e:
        attention_extractor = EnhancedAttentionExtractor("distilbert-base-uncased", model_type="transformer")
        print("   ✓ Using transformer model for seed-focused analysis")
    
    # Seed instructions with focus on the original concept
    seed_focused_prompt = f"""
    <seed_concept>{seed_concept}</seed_concept>
    <condensation_rule>
    Extract and preserve ONLY content that directly relates to, explains, or implements the seed concept.
    REMOVE: tangential discussions, implementation details not core to the concept, verbose explanations
    KEEP: mathematical foundations, core algorithms, architectural insights, key breakthroughs
    CONDENSE: multi-paragraph explanations into single, precise statements
    </condensation_rule>
    """
    
    seeded_text = seed_focused_prompt + conversation_text
    
    # Create context windows with seed focus
    print("📦 Creating seed-focused context windows...")
    windows = context_window.create_window(seeded_text)
    print(f"   Created {len(windows)} context windows\n")
    
    # Extract attention patterns with seed focus
    print("🧠 Extracting attention patterns...")
    attention_data = attention_extractor.extract_attention(windows)
    print(f"   Model: {attention_data['model_type']}")
    
    # Analyze for seed-relevant content
    analysis = attention_extractor.analyze_patterns(attention_data)
    
    # Create condensed chunks focused on seed
    print("✂️ Creating seed-focused chunks...")
    seed_relevant_chunks = []
    
    # Split the conversation into sentences
    sentences = conversation_text.split('. ')
    
    for sentence in sentences:
        # Check if sentence relates to the seed concept
        relevance_score = calculate_seed_relevance(sentence, seed_concept)
        
        if relevance_score > 0.3:  # Threshold for relevance
            # Condense the sentence if it's too verbose
            condensed = condense_sentence(sentence)
            seed_relevant_chunks.append({
                'content': condensed,
                'relevance': relevance_score,
                'type': classify_content_type(sentence)
            })
    
    print(f"   Original sentences: {len(sentences)}")
    print(f"   Seed-relevant chunks: {len(seed_relevant_chunks)}")
    
    # Sort by relevance and organize by type
    seed_relevant_chunks.sort(key=lambda x: x['relevance'], reverse=True)
    
    # Group by content type
    organized_content = {
        'theory': [],
        'mathematical': [],
        'implementation': [],
        'architectural': []
    }
    
    for chunk in seed_relevant_chunks:
        content_type = chunk['type']
        if content_type in organized_content:
            organized_content[content_type].append(chunk)
    
    # Generate condensed document
    print("\n📋 Generating seed-focused condensed document...")
    
    condensed_doc = generate_condensed_document(seed_concept, organized_content)
    
    # Save the condensed version
    output_filename = f"/workspace/results/seed_condensed_layered_partitions_{get_timestamp()}.md"
    
    with open(output_filename, 'w') as f:
        f.write(condensed_doc)
    
    print(f"   💾 Saved condensed document: {output_filename}")
    
    # Show statistics
    original_length = len(conversation_text)
    condensed_length = len(condensed_doc)
    compression_ratio = (original_length - condensed_length) / original_length * 100
    
    print(f"\n📊 Condensation Statistics:")
    print(f"   Original length: {original_length:,} characters")
    print(f"   Condensed length: {condensed_length:,} characters") 
    print(f"   Compression ratio: {compression_ratio:.1f}%")
    print(f"   Seed relevance preserved: {len(seed_relevant_chunks)} key concepts")
    
    return output_filename

def calculate_seed_relevance(text, seed_concept):
    """Calculate how relevant a text snippet is to the seed concept"""
    text_lower = text.lower()
    seed_lower = seed_concept.lower()
    
    # Key terms from the seed concept
    key_terms = ['layered', 'partition', 'context', 'window', 'interpolate', 'graph', 'overlap', 'percolation']
    
    relevance_score = 0
    
    # Direct mention of seed concept
    if seed_lower in text_lower:
        relevance_score += 1.0
    
    # Count key terms
    for term in key_terms:
        if term in text_lower:
            relevance_score += 0.2
    
    # Boost for mathematical/technical content related to the concept
    technical_terms = ['algorithm', 'matrix', 'tensor', 'attention', 'threshold', 'boundary', 'segmentation']
    for term in technical_terms:
        if term in text_lower:
            relevance_score += 0.1
    
    return min(relevance_score, 1.0)  # Cap at 1.0

def classify_content_type(text):
    """Classify content as theory, mathematical, implementation, or architectural"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['theorem', 'proof', 'mathematical', 'equation', 'formula']):
        return 'mathematical'
    elif any(word in text_lower for word in ['class', 'function', 'code', 'implementation', 'python']):
        return 'implementation'
    elif any(word in text_lower for word in ['architecture', 'system', 'design', 'structure', 'component']):
        return 'architectural'
    else:
        return 'theory'

def condense_sentence(sentence):
    """Condense a verbose sentence into a more concise form"""
    # Remove filler phrases
    filler_phrases = [
        'I think', 'it seems', 'basically', 'essentially', 'you know',
        'in other words', 'to put it simply', 'as I mentioned'
    ]
    
    condensed = sentence
    for phrase in filler_phrases:
        condensed = condensed.replace(phrase, '')
    
    # Remove extra whitespace
    condensed = ' '.join(condensed.split())
    
    return condensed

def generate_condensed_document(seed_concept, organized_content):
    """Generate the final condensed document focused on the seed"""
    
    doc = f"""# Condensed Analysis: {seed_concept.title()}

*Condensed from layered context graph conversation*
*Focus: Core concepts and implementations only*

## The Core Concept

{seed_concept} represents a mathematical approach to context management where:

"""
    
    # Add theory section
    if organized_content['theory']:
        doc += "## Theoretical Foundation\n\n"
        for chunk in organized_content['theory'][:5]:  # Top 5 most relevant
            doc += f"• {chunk['content']}\n"
        doc += "\n"
    
    # Add mathematical section
    if organized_content['mathematical']:
        doc += "## Mathematical Basis\n\n"
        for chunk in organized_content['mathematical'][:3]:  # Top 3 most relevant
            doc += f"• {chunk['content']}\n"
        doc += "\n"
    
    # Add architectural insights
    if organized_content['architectural']:
        doc += "## Architecture\n\n"
        for chunk in organized_content['architectural'][:4]:  # Top 4 most relevant
            doc += f"• {chunk['content']}\n"
        doc += "\n"
    
    # Add implementation notes
    if organized_content['implementation']:
        doc += "## Implementation Notes\n\n"
        for chunk in organized_content['implementation'][:3]:  # Top 3 most relevant
            doc += f"• {chunk['content']}\n"
        doc += "\n"
    
    # Add key insights
    doc += """## Key Insights

1. **Percolation Theory**: 15-30% overlap between windows ensures connectivity
2. **Attention Guidance**: Natural language rules can bias attention head behavior  
3. **Graph Emergence**: Knowledge graphs naturally emerge from layered partitions
4. **Bidirectional Flow**: Information flows both forward and backward in time

## Practical Applications

- Transcript processing and condensation
- Jupyter notebook refactoring
- Document reorganization
- Knowledge graph construction

---

*This condensed version preserves the essential insights while removing verbose explanations and tangential discussions.*
"""
    
    return doc

def get_timestamp():
    """Get current timestamp for filenames"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if __name__ == "__main__":
    condense_around_seed()
