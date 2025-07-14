# Project AURA: Adaptive Universal Reorganization Architecture

## 1. Core Philosophy: The "Tape-to-Graph" Transformation

At its heart, this project treats any linear document—be it a transcript, a codebase, or a research paper—as a one-dimensional **"tape"** of information. While easy to produce, this linear format obscures the complex, non-linear relationships between the concepts within it. Our system transforms this simple tape into a rich, multi-dimensional **knowledge graph**.

The process is as follows:

1.  **Disassembly (Tape to Nodes):** The continuous tape is first segmented into discrete, semantic "nodes." Each node represents a coherent idea, a code block, or a conversational turn. This is not an arbitrary split; it is a meaning-preserving discretization.
2.  **Reconstruction (Nodes to Graph):** We then establish relational edges between these nodes. These edges represent dependencies, explanations, contradictions, or temporal sequences, creating a true network of knowledge.
3.  **Reassembly (Graph to Purpose-Driven Documents):** The knowledge graph is not the final output. It is a powerful intermediate representation. From this single graph, we can reassemble the information into countless new "tapes," each optimized for a specific purpose (e.g., a tutorial, an executive summary, a technical reference).

## 2. The Mathematical Justification: Percolation Theory and Optimal Chunking

A critical step in the "Tape-to-Nodes" transformation is determining the ideal size and overlap of our initial text chunks. We ground our approach in **Percolation Theory**.

Imagine our document's concepts as a physical medium. For information to "percolate" or flow from one end of the document to the other, the chunks (nodes) must be sufficiently connected.

-   **Too little overlap (<15%):** The chunks are disconnected islands of meaning. The system can understand local concepts but fails to form a "big picture" view. No "giant connected component" of knowledge emerges.
-   **Too much overlap (>30%):** The system becomes computationally inefficient and redundant. The connections are trivially obvious, and we lose the ability to identify meaningful, non-local relationships.

The **critical threshold** lies in the **15-30% overlap range**. At this density, a phase transition occurs. The graph of nodes becomes globally connected, allowing insights and context to flow across the entire information space. This ensures that the meaning of a concept at the beginning of the tape can influence and be influenced by a concept at the end, enabling true retroactive understanding. This mathematically-grounded overlap is fundamental to our chunking strategy, ensuring the resulting knowledge graph is both coherent and comprehensive.

## 3. Current Implementation: Disassembly and Reassembly Rules

The system uses **separate disassembly and reassembly rules** to first break down text optimally, then reconstruct it with different organizational principles.

### 3.1. DISASSEMBLY RULES (Breaking Down)
**Location**: `layered-context-graph/src/partitioning/partition_manager.py`

#### Rule K1: Initial Disassembly Rules
**Method**: `_apply_disassembly_rules()`
- **Semantic boundaries**: Split at paragraphs, major topic shifts
- **Attention clusters**: Use attention patterns to find natural breaks
- **Percolation thresholds**: Apply percolation theory boundaries
- **Instruction markers**: Split at special tokens (QWQ_REASONING, etc.)

#### Rule K2: Iterative Segmentation Rules
**Method**: `_iterative_segmentation()`
- **Target**: Continue segmentation until average segment length ≈ 400 characters
- **Max rounds**: Up to 5 rounds of refinement
- **Convergence**: Stop when segments reach optimal size or no change occurs

#### Rule K3: Round-Specific Splitting Criteria
**Method**: `_split_by_round_criteria()`
- **Round 1**: `_split_by_semantic_boundaries()` - Paragraphs, sentences
- **Round 2**: `_split_by_syntactic_boundaries()` - Clauses, phrases  
- **Round 3**: `_split_by_instruction_markers()` - Special tokens
- **Round 4+**: `_split_by_character_count()` - Fallback fixed-size

#### Rule K4: Merging Criteria (when segments too small)
**Method**: `_merge_segments()`
- **Target**: Combine small segments up to 1.2x target length
- **Preserve**: Semantic coherence during merging
- **Strategy**: Greedy combination with overlap management

### 3.2. REASSEMBLY RULES (Reconstructing) 
**Location**: `layered-context-graph/src/graph/graph_reassembler.py`

#### Rule G1: Reconstruction Analysis
**Method**: `_analyze_optimal_segments()`
- **Input**: Final optimal segments from disassembly
- **Analysis**: Categorize segment types, themes, importance
- **Output**: Segment analysis metadata for reconstruction

#### Rule G2: Reconstruction Rules (Different from Disassembly)
**Method**: `_apply_reconstruction_rules()`
- **Importance ordering**: Organize by conceptual significance
- **Conceptual clustering**: Group related concepts together
- **Flow optimization**: Arrange for logical reading flow
- **Layered organization**: Create hierarchical structure

#### Rule G3: Content Generation Rules
**Method**: `_generate_reorganized_content()`
- **Layer-based output**: Organize content into conceptual layers
- **Flow optimization**: Ensure smooth transitions between concepts
- **Coherence preservation**: Maintain semantic relationships
- **Readable formatting**: Add headers, structure, navigation

### 3.3. PIPELINE FLOW

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

## 4. API-Based Implementation

The current implementation uses a robust, fallback approach that relies on standard LLM API calls rather than direct attention head access.

### 4.1. `api_processor.py`
Builds a knowledge graph from a document using only standard LLM API calls.

### 4.2. `graph_utils.py`
Merges highly similar nodes and prunes weak edges to clean the graph. It also classifies nodes as KEEP, DELETE, or TRACK based on graph metrics and content.

### 4.3. `condenser.py`
The main user-facing class that orchestrates the entire process of transforming a transcript into various structured, condensed outputs.

## 5. Future Work and Aspirational Architecture

This section outlines the planned enhancements and the more advanced, attention-based architecture that the project is working towards.

### 5.1. Multi-Round Annotation
The system is designed to support a multi-round annotation process where a single base graph is enriched with multiple layers of analysis.

- **Base Graph**: Created once from clean semantic chunks.
- **Annotation Layers**: Multiple analysis rounds add metadata to the same nodes and edges.
- **Layer Types**:
    - **Syntactic Layer**: Grammar, POS tags, linguistic structure.
    - **Semantic Layer**: Topics, concepts, meaning relationships.
    - **Pragmatic Layer**: Intent, discourse, communicative purpose.

### 5.2. Attention-Driven Graph Creation
A more advanced implementation will use transformer attention patterns to directly guide the graph construction process.

- **Principle**: Attention mechanisms slice existing text, never recreate it.
- **Process**: The transformer analyzes semantic chunks and builds the knowledge graph directly.
- **Output**: A pure graph structure with original content preserved in the nodes and attention patterns as metadata.

### 5.3. Scaffold-Guided Reconstruction
The reassembly process can be enhanced by using the original document's structure as a scaffold.

- **Scaffold**: The original document structure is used as a template.
- **Method**: Graph content is mapped back to the original organization.
- **Benefits**: This maintains the natural flow of the document while incorporating the insights from the graph.

### 5.4. Implementation Plan for Advanced Features
- **Instruction Seeder**: A module to insert natural language instructions into text to guide attention heads.
- **Attention-Based Graph Builder**: A module to build graphs directly from attention patterns.
- **CUDA Optimization**: Add GPU acceleration for performance improvements.
