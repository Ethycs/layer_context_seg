# Complete Project Architecture with All Functions

## Master Architecture Diagram

```mermaid
graph TB
    subgraph "Entry Points"
        IT[Input Text]
        MP[master_processor.py<br/>Class: FullMasterProcessor<br/>Functions: main, process_text, process_single_pass]
    end

    subgraph "Configuration"
        MC[master_config.py<br/>Classes: DisassemblyConfig, AssemblyConfig,<br/>ReconstructionConfig, SpectralConfig, GraphConfig<br/>Functions: get_config, get_rule_set]
        PR[processing_rules.py<br/>Functions: get_processing_rules,<br/>get_conversation_rules]
        DC[load_demo_content.py<br/>Function: get_demo_content]
    end

    subgraph "Attention Models"
        AE[attention_extractor_qwq.py<br/>Class: EnhancedAttentionExtractor<br/>Function: create_attention_extractor]
        MAE[models/attention_extractor.py<br/>Class: EnhancedAttentionExtractor<br/>Functions: create_attention_extractor,<br/>extract_attention_for_tape_splitting]
        QWQ[qwq_model.py<br/>Class: QwQModel<br/>Functions: load_model, extract_attention,<br/>generate_text, compute_embeddings]
        OE[ollama_extractor.py<br/>Class: OllamaModelExtractor<br/>Functions: extract_attention, generate_text,<br/>compute_embeddings]
    end

    subgraph "Partitioning"
        PM[partition_manager.py<br/>Class: PartitionManager<br/>Functions: partition, create_optimal_segments]
        CPM[content_preserving_partition_manager.py<br/>Class: ContentPreservingPartitionManager<br/>Functions: partition, preserve_content_structure]
        FPM[formatting_preserving_partition_manager.py<br/>Class: FormattingPreservingPartitionManager<br/>Functions: partition, preserve_formatting]
        CW[context_window.py<br/>Class: ContextWindow<br/>Functions: create_windows, get_overlap_indices]
        IS[instruction_seeder.py<br/>Class: InstructionSeeder<br/>Functions: seed_instructions, add_markers]
    end

    subgraph "Graph Processing"
        GP[processor.py<br/>Class: GraphProcessor<br/>Functions: process, build_graph]
        AGB[attention_graph_builder.py<br/>Class: AttentionGraphBuilder<br/>Functions: build_graph, create_nodes, create_edges]
        ED[edge_detector.py<br/>Class: EdgeDetector<br/>Functions: detect_edges, compute_similarity]
        KGM[knowledge_graph_manager.py<br/>Class: KnowledgeGraphManager<br/>Functions: condense_graph, classify_nodes,<br/>merge_similar_nodes]
        HGB[hierarchical_graph_builder.py<br/>Class: HierarchicalGraphBuilder<br/>Functions: build_hierarchy, rank_nodes,<br/>create_layers]
    end

    subgraph "Synthesis"
        GR[graph_reassembler.py<br/>Class: GraphReassembler<br/>Functions: reassemble, apply_strategy,<br/>organize_by_importance]
        TS[tape_synthesizer.py<br/>Class: TapeSynthesizer<br/>Functions: synthesize, generate_summary,<br/>generate_tutorial]
        LTS[llm_tape_synthesizer.py<br/>Class: LLMTapeSynthesizer<br/>Functions: synthesize_with_llm,<br/>generate_executive_summary]
        SOM[som_generator.py<br/>Class: SOM_DocumentGenerator<br/>Functions: generate_som, create_map,<br/>organize_concepts]
    end

    subgraph "Integration"
        LCG[main.py<br/>Class: LayeredContextGraph<br/>Function: main, process_document]
        OI[ollama_integration.py<br/>Classes: OllamaAttentionExtractor,<br/>LayeredContextGraphIntegration<br/>Function: demonstrate_integration]
    end

    subgraph "Utilities"
        LCP[llm_connection_pool.py<br/>Class: LLMConnectionPool<br/>Functions: get_connection, release_connection,<br/>create_connection]
        CU[cuda_utils.py<br/>Functions: setup_cuda, get_gpu_info,<br/>check_gpu_memory]
        DTS[demo_torch_spectral.py<br/>Functions: visualize_attention_and_clustering,<br/>demonstrate_linguistic_programming,<br/>show_mathematical_foundation, main]
    end

    IT --> MP
    DC --> MP
    MC --> MP
    PR --> MC
    
    MP --> PM
    MP --> AE
    AE --> MAE
    MAE --> QWQ
    MAE --> OE
    
    PM --> GP
    CPM --> PM
    FPM --> PM
    CW --> PM
    IS --> CW
    
    GP --> AGB
    AGB --> ED
    GP --> KGM
    KGM --> HGB
    
    HGB --> GR
    GR --> TS
    GR --> LTS
    GR --> SOM
    
    OE --> LCP
    QWQ --> LCP
    MP --> CU
    
    LCG --> GP
    OI --> MAE
    DTS --> MP
```

## All Functions by File

### Root Directory

**master_processor.py**
- `main()` - Entry point
- `process_text()` - Main processing function
- `process_single_pass()` - Single-pass processing

**master_config.py**
- `get_config()` - Get configuration for mode
- `get_rule_set()` - Get processing rules

**attention_extractor_qwq.py**
- `create_attention_extractor()` - Factory function

**demo_torch_spectral.py**
- `visualize_attention_and_clustering()`
- `demonstrate_linguistic_programming()`
- `show_mathematical_foundation()`
- `main()`

**load_demo_content.py**
- `get_demo_content()` - Load demo text

### Graph Processing

**processor.py**
- `process()` - Main graph processing
- `build_graph()` - Construct graph

**attention_graph_builder.py**
- `build_graph()` - Build attention-based graph
- `create_nodes()` - Create graph nodes
- `create_edges()` - Create graph edges

**edge_detector.py**
- `detect_edges()` - Find relationships
- `compute_similarity()` - Calculate similarity

**knowledge_graph_manager.py**
- `condense_graph()` - Merge similar nodes
- `classify_nodes()` - Categorize nodes
- `merge_similar_nodes()` - Combine duplicates

**hierarchical_graph_builder.py**
- `build_hierarchy()` - Create layers
- `rank_nodes()` - Importance ranking
- `create_layers()` - Organize by level

**graph_reassembler.py**
- `reassemble()` - Reconstruct document
- `apply_strategy()` - Apply reassembly strategy
- `organize_by_importance()` - Sort by significance

### Models

**attention_extractor.py**
- `create_attention_extractor()` - Factory
- `extract_attention_for_tape_splitting()` - Get attention

**qwq_model.py**
- `load_model()` - Load QwQ model
- `extract_attention()` - Get attention patterns
- `generate_text()` - Generate text
- `compute_embeddings()` - Get embeddings

**ollama_extractor.py**
- `extract_attention()` - Extract via Ollama
- `generate_text()` - Generate via Ollama
- `compute_embeddings()` - Embeddings via Ollama

**context_window.py**
- `create_windows()` - Make text windows
- `get_overlap_indices()` - Find overlaps

**instruction_seeder.py**
- `seed_instructions()` - Add instructions
- `add_markers()` - Insert markers

### Partitioning

**partition_manager.py**
- `partition()` - Main partitioning
- `create_optimal_segments()` - Optimize segments

**content_preserving_partition_manager.py**
- `partition()` - Partition preserving content
- `preserve_content_structure()` - Keep structure

**formatting_preserving_partition_manager.py**
- `partition()` - Partition preserving format
- `preserve_formatting()` - Keep formatting

### Synthesis

**tape_synthesizer.py**
- `synthesize()` - Generate output
- `generate_summary()` - Create summary
- `generate_tutorial()` - Create tutorial

**llm_tape_synthesizer.py**
- `synthesize_with_llm()` - LLM synthesis
- `generate_executive_summary()` - Exec summary

**som_generator.py**
- `generate_som()` - Create SOM
- `create_map()` - Build concept map
- `organize_concepts()` - Arrange concepts

### Integration

**main.py**
- `main()` - Alternative entry
- `process_document()` - Process doc

**ollama_integration.py**
- `demonstrate_integration()` - Demo function

### Utilities

**llm_connection_pool.py**
- `get_connection()` - Get LLM connection
- `release_connection()` - Release connection
- `create_connection()` - New connection

**cuda_utils.py**
- `setup_cuda()` - Initialize CUDA
- `get_gpu_info()` - GPU information
- `check_gpu_memory()` - Memory check

**processing_rules.py**
- `get_processing_rules()` - Get rules
- `get_conversation_rules()` - Conversation rules