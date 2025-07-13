# Disassembly and Reassembly Rules in Layered Context Graph

## 🔍 Overview
The system uses **separate disassembly and reassembly rules** to first break down text optimally, then reconstruct it with different organizational principles.

## 📂 DISASSEMBLY RULES (Breaking Down)
**Location**: `/src/partitioning/partition_manager.py`

### Rule K1: Initial Disassembly Rules
**Method**: `_apply_disassembly_rules()`
- **Semantic boundaries**: Split at paragraphs, major topic shifts
- **Attention clusters**: Use attention patterns to find natural breaks
- **Percolation thresholds**: Apply percolation theory boundaries
- **Instruction markers**: Split at special tokens (QWQ_REASONING, etc.)

### Rule K2: Iterative Segmentation Rules
**Method**: `_iterative_segmentation()`
- **Target**: Continue segmentation until average segment length ≈ 400 characters
- **Max rounds**: Up to 5 rounds of refinement
- **Convergence**: Stop when segments reach optimal size or no change occurs

### Rule K3: Round-Specific Splitting Criteria
**Method**: `_split_by_round_criteria()`
- **Round 1**: `_split_by_semantic_boundaries()` - Paragraphs, sentences
- **Round 2**: `_split_by_syntactic_boundaries()` - Clauses, phrases  
- **Round 3**: `_split_by_instruction_markers()` - Special tokens
- **Round 4+**: `_split_by_character_count()` - Fallback fixed-size

### Rule K4: Merging Criteria (when segments too small)
**Method**: `_merge_segments()`
- **Target**: Combine small segments up to 1.2x target length
- **Preserve**: Semantic coherence during merging
- **Strategy**: Greedy combination with overlap management

## 🏗️ REASSEMBLY RULES (Reconstructing) 
**Location**: `/src/graph/graph_reassembler.py`

### Rule G1: Reconstruction Analysis
**Method**: `_analyze_optimal_segments()`
- **Input**: Final optimal segments from disassembly
- **Analysis**: Categorize segment types, themes, importance
- **Output**: Segment analysis metadata for reconstruction

### Rule G2: Reconstruction Rules (Different from Disassembly)


**Method**: `_apply_reconstruction_rules()`
- **Importance ordering**: Organize by conceptual significance
- **Conceptual clustering**: Group related concepts together
- **Flow optimization**: Arrange for logical reading flow
- **Layered organization**: Create hierarchical structure

### Rule G3: Content Generation Rules
**Method**: `_generate_reorganized_content()`
- **Layer-based output**: Organize content into conceptual layers
- **Flow optimization**: Ensure smooth transitions between concepts
- **Coherence preservation**: Maintain semantic relationships
- **Readable formatting**: Add headers, structure, navigation

## 🔄 PIPELINE FLOW

```
Input Text
    ↓
[DISASSEMBLY PHASE]
    ↓
Rule K1: Initial split by semantic/attention boundaries
    ↓
Rule K2: Iterative refinement (Rounds 1-5)
    ├─ Rule K3: Round-specific criteria
    └─ Rule K4: Merging when needed
    ↓
Optimal Segments (avg ~400 chars)
    ↓
[REASSEMBLY PHASE]  
    ↓
Rule G1: Analyze optimal segments
    ↓
Rule G2: Apply reconstruction rules
    ├─ Importance ordering
    ├─ Conceptual clustering  
    ├─ Flow optimization
    └─ Layered organization
    ↓
Rule G3: Generate reorganized content
    ↓
Final Reorganized Output
```

## 🎯 KEY DIFFERENCES: K vs G Rules

| Aspect | Disassembly Rules (K) | Reassembly Rules (G) |
|--------|----------------------|----------------------|
| **Goal** | Break into optimal chunks | Organize for readability |
| **Criteria** | Attention, syntax, semantics | Importance, flow, hierarchy |
| **Process** | Iterative segmentation | Holistic reconstruction |
| **Metrics** | Segment length, boundaries | Coherence, organization |
| **Output** | List of optimal segments | Structured, layered content |

## 📍 Current Implementation Status

### ✅ IMPLEMENTED
- **K1**: Basic disassembly rules in `partition_manager.py`
- **K2**: Iterative segmentation with target length
- **K3**: Round-specific splitting criteria  
- **G1**: Segment analysis in `graph_reassembler.py`
- **G2**: Basic reconstruction framework

### 🚧 PARTIALLY IMPLEMENTED  
- **K4**: Merging logic (basic implementation)
- **G3**: Content generation (needs enhancement)

### ❌ NEEDS ENHANCEMENT
- **Attention-based splitting**: More sophisticated attention pattern analysis
- **Percolation boundaries**: Full percolation theory implementation
- **Flow optimization**: Advanced content flow algorithms
- **Layer hierarchy**: More sophisticated layering logic

## 🔧 Configuration

### Disassembly Configuration
```python
# In partition_manager.py
self.disassembly_rules = {
    'semantic_boundaries': True,      # Rule K1
    'attention_clusters': True,       # Rule K1  
    'percolation_thresholds': True,   # Rule K1
    'instruction_markers': True       # Rule K1
}
self.target_segment_length = 400     # Rule K2
self.max_rounds = 5                  # Rule K2
```

### Reassembly Configuration  
```python
# In graph_reassembler.py
self.reassembly_rules = {
    'importance_ordering': True,      # Rule G2
    'conceptual_clustering': True,    # Rule G2
    'flow_optimization': True,        # Rule G2  
    'layered_organization': True      # Rule G2
}
```

## 🎯 Next Steps for Enhancement

1. **Enhance attention-based disassembly** in `Rule K1`
2. **Improve percolation boundary detection** in `Rule K1`  
3. **Sophisticated flow optimization** in `Rule G2`
4. **Better layer hierarchy generation** in `Rule G3`
5. **Content quality metrics** for both phases
