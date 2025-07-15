# Refactoring Plan for Project AURA

## 1. Introduction

Based on a review of the current architecture and the project philosophy outlined in `README.md`, this document proposes a refactoring plan to simplify the application, improve maintainability, and more closely align the code with the "Tape-to-Graph" vision.

The core goal is to consolidate responsibilities into more cohesive modules, clarifying the main data processing pipeline without sacrificing functionality.

## 2. Proposed Refactoring Steps

### Step 1: Introduce a Unified `GraphProcessor` Module

-   **Problem**: The logic for creating and manipulating the knowledge graph is currently spread across multiple classes (`AttentionGraphBuilder`, `KnowledgeGraphManager`, `HierarchicalGraphBuilder`), making the main pipeline in `master_processor.py` complex and procedural.
-   **Solution**:
    1.  Create a new file: `layered-context-graph/src/graph/processor.py`.
    2.  Inside this file, define a new class `GraphProcessor`.
    3.  This class will encapsulate the entire graph lifecycle, from initial construction to final hierarchical structuring. It will have a main public method, `process(segments)`, which orchestrates the following internal steps:
        -   Building the initial graph from segments (using `AttentionGraphBuilder`).
        -   Classifying nodes and condensing the graph (using `KnowledgeGraphManager`).
        -   Transforming the flat graph into a hierarchical structure (using `HierarchicalGraphBuilder`).
-   **Benefit**: This change will dramatically simplify `master_processor.py`. Instead of orchestrating three separate graph-related objects, it will make a single, clear call to the `GraphProcessor`.

### Step 2: Consolidate Edge Detection Logic

-   **Problem**: The `AttentionGraphBuilder` depends on two separate edge detectors (`AttentionBasedEdgeDetector` and `EnhancedEdgeDetector`), which increases complexity.
-   **Solution**:
    1.  Create a new, consolidated `EdgeDetector` class in a new file or by merging into one of the existing files.
    2.  This class will contain the methods from both original detectors (e.g., `detect_from_attention`, `detect_from_rules`).
    3.  The `AttentionGraphBuilder` will be updated to use this single `EdgeDetector`.
    4.  The original, now redundant, edge detector files will be deleted.
-   **Benefit**: This reduces the number of files in the `graph` module and simplifies the dependency graph.

### Step 3: Simplify the Main Pipeline in `master_processor.py`

-   **Problem**: The `_process_single_pass` method is currently a long, procedural script that details every step of the process.
-   **Solution**: Refactor the method to be a high-level, easy-to-read sequence of calls to the primary components. The new pipeline will look like this:

    ```python
    # 1. Disassembly Phase (Tape to Nodes)
    segments = self.partitioner.partition(text)
    
    # 2. Reconstruction Phase (Nodes to Graph)
    graph_data = self.graph_processor.process(segments)
    
    # 3. Reassembly Phase (Graph to Document)
    reassembled_output = self.reassembler.reassemble(
        graph_data.nodes, 
        graph_data.edges, 
        strategy='layered_assembly'
    )
    ```
-   **Benefit**: This makes the overall architecture much clearer and directly mirrors the three-phase philosophy described in `README.md`.

### Step 4: Unify Configuration

-   **Problem**: Configuration settings are currently split between `master_config.py` and `config/graph_config.py`.
-   **Solution**: Consolidate all configuration into a single, unified system managed from `master_config.py`. The settings from `graph_config.py` will be merged into the main configuration dictionary, and the redundant file will be removed.
-   **Benefit**: This creates a single source of truth for all settings, making the system easier to configure and manage.

## 3. Proposed Architecture Diagram

This diagram illustrates the simpler, more modular architecture that will result from this refactoring.

```mermaid
graph TD
    subgraph "Entry & Configuration"
        A[Input Text] --> MP[master_processor.py];
        UnifiedConfig[master_config.py] --> MP;
    end

    subgraph "Core Components"
        MP -- instantiates --> Part[partitioning/partition_manager.py];
        MP -- instantiates --> GP[graph/processor.py];
        MP -- instantiates --> GR[graph/graph_reassembler.py];
    end

    subgraph "Simplified Pipeline"
        MP -- "1. partition (Disassembly)" --> Part;
        Part --> Segments;
        Segments --> GP;
        GP -- "2. process (Reconstruction)" --> HierarchicalGraph;
        HierarchicalGraph --> GR;
        GR -- "3. reassemble (Reassembly)" --> FinalOutput[Final Output Document];
    end

    subgraph "GraphProcessor Internals"
        GP -- uses --> AGB[attention_graph_builder.py];
        AGB -- uses --> ED[EdgeDetector (Consolidated)];
        GP -- uses --> KGM[knowledge_graph_manager.py];
        GP -- uses --> HGB[hierarchical_graph_builder.py];
    end

    style MP fill:#f9f,stroke:#333,stroke-width:2px
    style GP fill:#ccf,stroke:#333,stroke-width:2px
```

By implementing this plan, the codebase will become more modular, maintainable, and easier to extend, while fully aligning with the project's architectural vision.
