# Layered Context Graph System - Architecture Guide v2.0

## Core Philosophy

The Layered Context Graph System follows a **Source-First, Scaffold-Guided** approach where:
- **Fluff removal happens at source** during initial partitioning
- **Transformers create graphs directly** from semantic chunks
- **Original document serves as reconstruction scaffold** for coherent reassembly
- **Single-pass processing** eliminates redundant partitioning and deduplication

## Revolutionary Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Text    │───▶│  Semantic Windows │───▶│ Transformer     │
│  (Original Doc) │    │  + Fluff Removal │    │ Graph Creation  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                                               │
        │ (serves as scaffold)                          ▼
        │                                    ┌─────────────────┐
        └──────────────────────────────────▶ │ Scaffold-Guided │
                                             │ Reconstruction  │
                                             └─────────────────┘
```

## Key Innovations

### 1. Fluff Removal at Source
- **Where**: During semantic window creation in `ContextWindow._create_semantic_windows()`
- **What**: Eliminates filler words, duplicate sentences, repetitive phrases
- **Why**: Prevents transformers from processing redundant content
- **Result**: Cleaner graphs, no post-processing deduplication needed

### 2. Transformer-Driven Graph Creation
- **Principle**: Attention mechanisms slice existing text, never recreate it
- **Process**: Transformer analyzes semantic chunks and builds knowledge graph
- **No Redundancy**: Single partitioning step, no multi-round processing
- **Output**: Pure graph structure with original content preserved

### 3. Scaffold-Guided Reconstruction
- **Scaffold**: Original document structure used as template
- **Method**: Graph content mapped back to original organization
- **Benefits**: Maintains document flow while incorporating graph insights
- **Result**: Coherent reconstruction that feels natural

## System Components

### 1. Semantic Processing Layer (`src/models/`)

#### ContextWindow (Enhanced with Fluff Removal)
- **Purpose**: Creates semantic chunks while eliminating redundancy at source
- **Key Innovation**: `_remove_fluff()` method processes content before transformer
- **Fluff Patterns Removed**:
  - Filler words (um, uh, like, basically, actually, literally)
  - Duplicate sentences within paragraphs
  - Repetitive phrases and patterns
  - Excessive whitespace and punctuation
- **Preservation**: Code blocks and technical content protected
- **Result**: Clean semantic windows ready for transformer processing

#### AttentionExtractor (Streamlined)
- **Purpose**: Extracts attention patterns without text modification
- **Key Principle**: **Analysis only, no content transformation**
- **Methods**:
  - `extract_attention(text)`: Returns attention metadata only
  - Original text content remains untouched in graph nodes
- **Transformer Models**: Works with both HuggingFace and Ollama GGUF models
- **Output**: Attention patterns as metadata, not content replacement

#### OllamaIntegration (Enhanced)
- **Purpose**: Integrates GGUF models for attention analysis
- **Key Features**:
  - Direct GGUF weight extraction
  - Static attention pattern analysis
  - QwQ 32B architecture support
- **Process**: Analyzes model weights to understand attention flow
- **Output**: Attention fingerprints for graph construction

### 2. Graph Construction Layer (`src/graph/`)

#### Knowledge Graph (Transformer-Built)
- **Purpose**: Constructed directly by transformer from semantic chunks
- **Process**:
  1. Transformer receives clean semantic windows
  2. Builds graph nodes with original content
  3. Creates edges based on attention relationships
  4. No post-processing modification of content
- **Node Structure**: Original text + attention metadata
- **Edge Weights**: Computed from attention similarity patterns

#### GraphReassembler (Scaffold-Guided)
- **Revolutionary Feature**: Uses original document as reconstruction scaffold
- **Process**:
  1. `create_document_scaffold()`: Maps original structure
  2. `scaffold_guided_hierarchy()`: Organizes graph content using scaffold
  3. `generate_content_using_scaffold()`: Fills scaffold with graph insights
  4. Maintains document flow while incorporating extracted knowledge
- **Result**: Natural-feeling reconstruction that preserves original organization

### 3. Deduplication Strategy (Conservative)

#### Single-Point Deduplication
- **Where**: Only in `GraphReassembler._deduplicate_nodes()`
- **Threshold**: Increased from 70% to 95% similarity for merging
- **Strategy**: Very conservative - only merge truly identical content
- **Substring Merging**: Disabled aggressive substring containment
- **Result**: Preserves unique content, eliminates only true duplicates
## Simplified Data Flow

### 1. Source Processing (Fluff Removal)
```
Raw Text → Semantic Windowing → Fluff Removal → Clean Chunks
```
- **Input**: Original document text
- **Process**: Split into semantic paragraphs, remove redundancy
- **Output**: Clean semantic chunks with original content preserved
- **Key**: No modification of meaningful content, only fluff elimination

### 2. Transformer Graph Construction
```
Clean Chunks → Attention Analysis → Direct Graph Creation
```
- **Input**: Clean semantic chunks from step 1
- **Process**: Transformer analyzes chunks and builds knowledge graph
- **Output**: Graph with nodes containing original content + attention metadata
- **Key**: Single pass, no redundant partitioning

### 3. Scaffold-Guided Reconstruction
```
Original Text + Knowledge Graph → Document Scaffold → Enhanced Reconstruction
```
- **Scaffold**: Original document structure and organization
- **Content**: Enhanced insights from knowledge graph
- **Process**: Map graph content back to original structure
- **Output**: Coherent document with preserved flow and enhanced insights

## Configuration (Simplified)

### Core Settings
```python
CONFIG = {
    # Source processing
    'window_size': 8192,  # Words per semantic chunk
    'fluff_removal': True,  # Enable source-level cleanup
    
    # Graph construction  
    'single_pass_processing': True,  # No redundant partitioning
    'preserve_original_content': True,  # Never modify source text
    
    # Reconstruction
    'use_document_scaffold': True,  # Use original as template
    'conservative_deduplication': True,  # 95% similarity threshold
    
    # Model settings
    'model_type': 'transformer',  # or 'ollama'
    'model_name': 'distilbert-base-uncased'  # or 'qwq'
}
```

## Mathematical Foundation (Updated)

### Fluff Removal Metrics
```python
# Redundancy reduction at source
def fluff_removal_efficiency(original_length, cleaned_length):
    return (original_length - cleaned_length) / original_length

# Content preservation (should be high)
def content_preservation_ratio(meaningful_content_kept, total_meaningful):
    return meaningful_content_kept / total_meaningful
```

### Scaffold Alignment
```python
# How well reconstruction follows original structure
def scaffold_alignment_score(original_structure, reconstructed_structure):
    return structural_similarity(original_structure, reconstructed_structure)

# Information enhancement from graph
def information_enhancement_ratio(graph_insights, original_insights):
    return graph_insights / original_insights
```
  - Dependency analysis between cells
  - Metadata preservation

#### Visualization
- **Purpose**: Provides visual debugging and analysis tools
- **Capabilities**:
  - Attention pattern heatmaps
  - Graph structure visualization
  - Partition boundary display
  - Percolation metrics dashboard

## Mathematical Foundation

### Percolation Theory Application

The system uses percolation theory to ensure optimal connectivity:

```python
# Critical overlap ratio for connectivity
overlap_ratio = 0.25  # 25% overlap ensures percolation

# Percolation threshold calculation
def calculate_threshold(graph_density, node_count):
    return 1.0 / (node_count - 1)  # Critical threshold for connectivity
```

### Attention Pattern Analysis

Attention weights are converted to semantic relationships:

```python
# Attention fingerprint calculation
fingerprint = attention_weights.mean(dim=1)  # Average across heads

# Similarity calculation for edge weights
edge_weight = cosine_similarity(fingerprint_i, fingerprint_j)
```

## Data Flow

1. **Input Processing**:
   ```
   Raw Text → Instruction Seeding → Context Window Creation
   ```

2. **Partition Creation**:
   ```
   Context Window → Attention Extraction → Spectral Clustering → Partitions
   ```

3. **Graph Construction**:
   ```
   Partitions → Node Creation → Edge Calculation → Knowledge Graph
   ```

4. **Classification & Management**:
   ```
   Knowledge Graph → Node Classification → Graph Optimization → Output
   ```

## Configuration

### Default Parameters
```python
CONFIG = {
    'window_size': 8192,
    'overlap_ratio': 0.25,
    'instruction_density': 0.15,
    'percolation_threshold': 0.593,  # 2D lattice critical point
    'attention_heads': 12,
    'embedding_dim': 768
}
```

### Environment Setup
```bash
# Required packages
torch>=1.9.0
transformers>=4.0.0
torch-geometric>=2.0.0
networkx>=2.6.0
matplotlib>=3.5.0
jupyter>=1.0.0
```

## Performance Considerations

### Memory Management
- **Context Window**: O(n) where n = window_size
- **Attention Storage**: O(h × n²) where h = attention_heads
- **Graph Storage**: O(V + E) where V = nodes, E = edges

### Optimization Strategies
1. **Attention Caching**: Store computed attention patterns
2. **Lazy Graph Construction**: Build graph incrementally
3. **Partition Pruning**: Remove low-confidence boundaries
4. **Memory-Mapped Storage**: For large graphs

## Testing Strategy

### Unit Tests
- Individual component functionality
- Mathematical correctness of percolation calculations
- Attention pattern extraction accuracy

### Integration Tests
- End-to-end pipeline processing
- Graph construction and reassembly
- Notebook parsing and refactoring

### Performance Tests
- Memory usage profiling
- Computational complexity validation
- Scalability testing with large contexts

## Future Extensions

### Planned Features
1. **Multi-Modal Support**: Images, code, equations
2. **Incremental Learning**: Dynamic graph updates
3. **Distributed Processing**: Multi-GPU graph construction
4. **Export Formats**: GraphML, Neo4j, RDF

### Research Directions
1. **Adaptive Partitioning**: Learning optimal boundaries
2. **Hierarchical Graphs**: Multi-scale representations
3. **Attention Ensemble**: Multiple model integration
4. **Real-time Processing**: Streaming context windows

## Usage Patterns

### Basic Usage
```python
from src.main import LayeredContextGraph

lcg = LayeredContextGraph()
text = "Your input text here..."
knowledge_graph = lcg.process(text)
reconstructed = lcg.reassemble(knowledge_graph, strategy='sequential')
```

### Advanced Configuration
```python
config = {
    'window_size': 16384,
    'overlap_ratio': 0.3,
    'instruction_types': ['<MATH>', '<CODE>', '<MEMORY>']
}
lcg = LayeredContextGraph(config)
```

This architecture provides a robust foundation for transforming context windows into knowledge graphs while maintaining theoretical soundness and practical efficiency.