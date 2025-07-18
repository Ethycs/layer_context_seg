# GAP Architecture Integration

## Overview

This document describes the integration of GAP (Graph-Aware Positional) architecture into the QwQ model for dual-level (token and node) processing. The implementation enables graph-aware attention mechanisms while maintaining compatibility with existing low-rank compression and windowing approaches.

## Architecture Components

### Core Philosophy

GAP introduces dual-level processing where:
- **Token Level**: Standard transformer attention on individual tokens
- **Node Level**: Graph-aware attention on aggregated segments/nodes
- **Cross-Level**: Information exchange between token and node representations

### Key Components

#### 1. Scatter Aggregation (`scatter_aggregation.py`)

**Purpose**: Implements scatter operations for token-to-node aggregation, the fundamental building block of GAP.

**Key Classes**:
- `TokenToNodeAggregator`: Aggregates token representations to node representations
- `NodeToTokenDistributor`: Distributes node information back to tokens
- `HierarchicalAggregator`: Handles multi-level hierarchical structures

**Core Operations**:
```python
# Scatter mean: averages token embeddings by node assignment
node_embeddings = scatter_mean(token_embeddings, token_to_node_mapping)

# Scatter sum: sums token embeddings by node
node_embeddings = scatter_sum(token_embeddings, token_to_node_mapping)

# Scatter max: max-pools token embeddings by node
node_embeddings, indices = scatter_max(token_embeddings, token_to_node_mapping)
```

#### 2. Dual-Level Processor (`dual_level_processor.py`)

**Purpose**: Implements the core GAP attention mechanism with dual-level processing.

**Key Classes**:
- `DualLevelAttentionProcessor`: Single attention layer with dual-level processing
- `DualLevelEncoder`: Complete encoder with multiple dual-level layers
- `DualLevelEncoderLayer`: Individual encoder layer
- `AdaptiveDualLevelProcessor`: Switches between token-only and dual-level based on input

**Processing Flow**:
1. **Token-Level Attention**: Standard multi-head attention on tokens
2. **Token-to-Node Aggregation**: Scatter operation to create node representations
3. **Node-Level Attention**: Graph-aware attention on nodes (with adjacency constraints)
4. **Cross-Level Attention**: Tokens attend to nodes
5. **Node-to-Token Distribution**: Distribute node information back to tokens
6. **Residual Connection**: Combine enhanced tokens with original

#### 3. Token-Segment Mapper (`token_segment_mapper.py`)

**Purpose**: Efficient mapping between tokenizer output and document segments.

**Key Classes**:
- `TokenSegmentMapper`: Creates token-to-segment alignments
- `TokenSegmentAlignment`: Stores alignment information
- `HierarchicalTokenMapper`: Handles multi-level hierarchical segments
- `SegmentGraphMapper`: Maps segments to graph nodes with adjacency

**Alignment Process**:
```python
# Create mapping from tokens to segments
alignment = mapper.create_mapping(
    tokens=tokenizer_output,
    segments=partition_manager_segments,
    text=original_text
)

# Result: TokenSegmentAlignment object with:
# - token_to_segment: [seq_len] tensor mapping tokens to segment IDs
# - segment_to_tokens: Dict mapping segment IDs to token lists
# - segment_boundaries: List of (start, end) token positions
```

#### 4. Enhanced QwQ Model (`qwq_model_dual_level.py`)

**Purpose**: Extends the original QwQ model with dual-level processing capabilities.

**Key Features**:
- **Backward Compatible**: Inherits from original `QwQModel`
- **Lazy Loading**: Dual-level components loaded only when needed
- **Window Processing**: Integrates with existing windowing approach
- **Compression Compatible**: Works with existing low-rank decomposition

**New Methods**:
```python
# Extract dual-level attention
results = model.extract_dual_level_attention(
    text=text,
    segments=segments,
    adjacency_matrix=adjacency,
    edge_types=edge_types,
    window_size=512,
    rank_ratio=0.1
)

# Segment with dual-level patterns
segments, graph = model.segment_with_dual_attention(
    text=text,
    rule="segment by topic",
    use_graph=True
)

# Get both token and node embeddings
embeddings = model.get_dual_level_embeddings(text, segments)
```

#### 5. Enhanced Attention Calculator (`attention_calculator_dual_level.py`)

**Purpose**: Extends attention analysis with dual-level pattern recognition.

**Key Features**:
- **Node Pattern Analysis**: Clustering, centrality, component analysis
- **Cross-Level Analysis**: Token-to-node attention patterns
- **Segment Cohesion**: Measures internal vs external attention
- **Dual-Level Boundaries**: Boundary detection using both levels

**Analysis Capabilities**:
```python
calculator = DualLevelAttentionCalculator()

# Process dual-level windows
calculator.process_dual_level_window(window_data)

# Get comprehensive results
results = calculator.get_dual_level_results()
# Results include:
# - Node patterns (clustering, centrality, density)
# - Cross-level patterns (consistency, strength)
# - Segment cohesion scores
# - Dual-level boundaries
```

## Integration with Existing System

### 1. Partition Manager Integration

The dual-level system seamlessly integrates with the existing `PartitionManager`:

```python
# Existing partition manager creates segments
pm = PartitionManager()
pm.partition(text, k_rules)

# Convert segments for dual-level processing
segments_list = []
for seg_id, segment in pm.segments.items():
    segments_list.append({
        'id': seg_id,
        'content': segment.content,
        'start_pos': segment.start_pos,
        'end_pos': segment.end_pos
    })

# Extract graph adjacency
adjacency = nx.to_numpy_array(pm.graph)

# Run dual-level analysis
dual_model = QwQModelDualLevel(model_path)
results = dual_model.extract_dual_level_attention(
    text=text,
    segments=segments_list,
    adjacency_matrix=adjacency
)
```

### 2. Low-Rank Compression Compatibility

GAP processing maintains full compatibility with existing compression:

```python
# Compression is applied AFTER graph constraints
# Processing order:
# 1. Compute raw attention scores
# 2. Add type embeddings (GAP enhancement)
# 3. Apply adjacency constraints (GAP enhancement)
# 4. Apply SVD low-rank decomposition (existing)
# 5. Apply softmax and compute output

# This gives multiplicative compression benefits:
# - Graph constraints naturally sparsify attention
# - SVD further compresses the sparse patterns
```

### 3. Windowed Processing Integration

The system handles windowed processing efficiently:

```python
# For each window:
# 1. Extract window tokens and text
# 2. Create window-specific token-to-segment mapping
# 3. Extract relevant portions of adjacency matrix
# 4. Process through dual-level attention
# 5. Apply existing compression (SVD)
# 6. Collect results

# Windows are processed independently and results merged
```

## Data Flow

### Forward Pass Data Flow

```
Input Text
    ↓
[Tokenization]
    ↓
Token Embeddings [batch, seq_len, d_model]
    ↓
[Token-to-Segment Mapping]
    ↓
┌─────────────────────┐    ┌─────────────────────┐
│   Token-Level       │    │   Node-Level        │
│   Attention         │    │   Attention         │
│                     │    │                     │
│ Q, K, V = Linear    │    │ Node Embeddings     │
│ Attention = QK^T/√d │    │ = Scatter(Tokens)   │
│ Output = Attention×V│    │                     │
└─────────────────────┘    │ Graph-Aware Attn    │
           ↓                │ (with adjacency)    │
    [Cross-Level            │                     │
     Attention]             └─────────────────────┘
           ↓                          ↓
    [Node-to-Token                    ↓
     Distribution]            [Back to Tokens]
           ↓                          ↓
    Enhanced Token Embeddings ←───────┘
           ↓
    [Residual Connection]
           ↓
    Final Output
```

### Attention Pattern Analysis

```
Raw Attention Patterns
    ↓
[Token-Level Analysis]
    ↓
├─ Attention Flow
├─ Entropy Analysis  
├─ Boundary Detection
└─ Pattern Extraction
    ↓
[Node-Level Analysis]
    ↓
├─ Graph Metrics (density, clustering)
├─ Centrality Analysis
├─ Component Detection
└─ Hub Identification
    ↓
[Cross-Level Analysis]
    ↓
├─ Attention Consistency
├─ Segment Cohesion
├─ Cross-Level Patterns
└─ Dual-Level Boundaries
    ↓
Comprehensive Results
```

## Performance Characteristics

### Memory Efficiency

- **Graph Sparsity**: Adjacency constraints naturally create sparse attention
- **SVD Compression**: Applied after graph constraints for double compression
- **Windowed Processing**: Maintains constant memory usage regardless of document length
- **Lazy Loading**: Dual-level components only loaded when needed

### Computational Complexity

- **Token-Level**: O(n²) where n = window_size (unchanged)
- **Node-Level**: O(m²) where m = num_segments (typically m << n)
- **Cross-Level**: O(n×m) for token-to-node attention
- **Aggregation**: O(n×m) for scatter operations

**Total Complexity**: O(n²) + O(m²) + O(n×m) ≈ O(n²) since m << n

### Compression Benefits

1. **Graph Sparsification**: Adjacency masks reduce effective attention space
2. **SVD Compression**: Low-rank decomposition on sparse patterns
3. **Segment Aggregation**: Reduces sequence length from n tokens to m segments
4. **Type Embeddings**: Lightweight scalar additions (minimal overhead)

## Usage Examples

### Basic Usage

```python
from layered_context_graph.src.models.qwq_model_dual_level import QwQModelDualLevel

# Initialize model
model = QwQModelDualLevel("./QwQ-32B", enable_dual_level=True)

# Define segments
segments = [
    {'id': '0', 'content': 'First paragraph...', 'start_pos': 0, 'end_pos': 100},
    {'id': '1', 'content': 'Second paragraph...', 'start_pos': 101, 'end_pos': 200}
]

# Optional: Define graph structure
adjacency = np.array([[0, 1], [1, 0]])  # Segments are connected

# Extract dual-level attention
results = model.extract_dual_level_attention(
    text=text,
    segments=segments,
    adjacency_matrix=adjacency,
    window_size=512,
    rank_ratio=0.1
)
```

### Advanced Usage with Calculator

```python
from layered_context_graph.src.graph.attention_calculator_dual_level import DualLevelAttentionCalculator

# Initialize calculator
calculator = DualLevelAttentionCalculator(
    rank_ratio=0.1,
    boundary_threshold=0.3,
    node_attention_weight=0.6
)

# Extract with analysis
results = model.extract_dual_level_attention(
    text=text,
    segments=segments,
    adjacency_matrix=adjacency,
    calculator=calculator
)

# Get comprehensive analysis
analysis = calculator.get_dual_level_results()
```

### Migration from Existing Code

```python
# Before (existing code)
from layered_context_graph.src.models.qwq_model import QwQModel
from layered_context_graph.src.graph.attention_calculator import AttentionCalculator

model = QwQModel("./QwQ-32B")
calculator = AttentionCalculator()
results = model.extract_attention(text, calculator=calculator)

# After (dual-level enhanced)
from layered_context_graph.src.models.qwq_model_dual_level import QwQModelDualLevel
from layered_context_graph.src.graph.attention_calculator_dual_level import DualLevelAttentionCalculator

model = QwQModelDualLevel("./QwQ-32B", enable_dual_level=True)
calculator = DualLevelAttentionCalculator()
results = model.extract_dual_level_attention(text, segments, calculator=calculator)
```

## Future Enhancements

### Planned Features

1. **Learnable Type Embeddings**: Fine-tune type embeddings on specific domains
2. **Dynamic Graph Construction**: Adapt graph structure during processing
3. **Multi-Scale Processing**: Different granularities for different layers
4. **GNN Integration**: Graph neural network layers for richer representations
5. **Attention Pattern Mining**: Discover recurring subgraph patterns

### Research Directions

1. **Optimal Graph Density**: Finding the sweet spot between connectivity and sparsity
2. **Adaptive Rank Selection**: Dynamic rank selection based on graph structure
3. **Cross-Domain Transfer**: How graph structures transfer across domains
4. **Scalability Studies**: Performance on very large documents
5. **Interpretability**: Understanding what node-level attention patterns mean

## Conclusion

The GAP architecture integration provides a powerful enhancement to the QwQ model, enabling sophisticated document understanding through dual-level processing. The implementation maintains backward compatibility while adding rich graph-aware capabilities that leverage the existing compression and windowing infrastructure.

The system successfully bridges the gap between token-level transformer processing and document-level structural understanding, opening new possibilities for document analysis, segmentation, and synthesis tasks.