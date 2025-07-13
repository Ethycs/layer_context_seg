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

### 1. Models Layer (`src/models/`)

#### AttentionExtractor
- **Purpose**: Extracts attention patterns from transformer models
- **Key Methods**:
  - `extract_attention(text)`: Returns attention weights and patterns
  - `analyze_patterns(attention_weights)`: Identifies semantic relationships
- **Dependencies**: PyTorch, transformers library

#### ContextWindow
- **Purpose**: Manages context window creation and partitioning
- **Key Features**:
  - Configurable window size (default: 8192 tokens)
  - Overlap management (15-30% for percolation threshold)
  - Dynamic boundary detection
- **Mathematical Foundation**: Based on percolation theory for optimal connectivity

#### KnowledgeGraph
- **Purpose**: Constructs and maintains the knowledge graph
- **Implementation**: PyTorch Geometric for graph operations
- **Key Features**:
  - Node classification (KEEP/DELETE/TRACK)
  - Edge weight calculation based on attention patterns
  - Graph traversal and query capabilities

#### Ollama Model Integration (`src/models/ollama_extractor.py`)

The system supports integration with Ollama's GGUF models for attention analysis:

- **OllamaModelExtractor**: Extracts models from Ollama's blob storage
- **GGUFConverter**: Converts GGUF format to PyTorch tensors
- **AttentionAnalyzer**: Analyzes static attention weights

##### Usage with Ollama Models

```python
from src.main import LayeredContextGraph

# Initialize with Ollama model
lcg = LayeredContextGraph(model_type="ollama", model_name="qwq32b")

# Process text using QWQ32B attention patterns
knowledge_graph = lcg.process(text)
```

##### Attention Pattern Extraction

Ollama models use static weight analysis rather than runtime attention:
- Layer fingerprints computed from Q, K, V projections
- Inter-layer similarity based on weight correlations
- Attention-based partitioning using weight clusters

##### Configuration

The Ollama integration is configured via `src/config.py`:
```python
OLLAMA_CONFIG = {
    'model_type': 'ollama',
    'model_name': 'qwq32b',
    'ollama_models_path': '/usr/share/ollama/.ollama/models',
    'cache_extracted_models': True,
    'extract_attention_only': True
}
```

### 2. Partitioning Layer (`src/partitioning/`)

#### InstructionSeeder
- **Purpose**: Injects instruction markers into text to bias attention patterns
- **Instruction Types**:
  - `<MATH>`: Mathematical content detection
  - `<DIALOGUE>`: Conversational patterns
  - `<MEMORY>`: Information to retain
  - `<TRACK>`: Items requiring follow-up
- **Strategy**: Random seeding with configurable density (10-20%)

#### PartitionManager
- **Purpose**: Creates and manages arbitrary partitions based on attention patterns
- **Algorithm**:
  1. Extract attention fingerprints per token
  2. Apply spectral clustering to group similar patterns
  3. Create soft boundaries with confidence scores
  4. Validate percolation connectivity

#### PercolationAnalyzer
- **Purpose**: Ensures graph connectivity using percolation theory
- **Key Metrics**:
  - Critical threshold calculation
  - Giant component detection
  - Information flow analysis
- **Optimization**: Maintains 15-30% overlap for optimal connectivity

### 3. Graph Layer (`src/graph/`)

#### GraphBuilder
- **Purpose**: Constructs the knowledge graph from partitions
- **Process**:
  1. Create nodes from text chunks
  2. Calculate edge weights using attention similarity
  3. Apply percolation rules for connectivity
  4. Optimize graph structure

#### NodeClassifier
- **Purpose**: Classifies nodes using LLM-driven decision making
- **Classifications**:
  - **KEEP**: High-value information for retention
  - **DELETE**: Redundant or low-value content
  - **TRACK**: Items requiring future attention
- **Implementation**: Uses instruction-seeded prompts for classification

#### GraphReassembler
- **Purpose**: Reconstructs coherent documents from the knowledge graph
- **Strategies**:
  - Sequential reassembly for narrative flow
  - Topic-based clustering for thematic organization
  - Attention-guided traversal for semantic coherence

### 4. Utilities Layer (`src/utils/`)

#### NotebookParser
- **Purpose**: Parses Jupyter notebooks and extracts structured content
- **Features**:
  - Cell type detection (code, markdown, output)
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