graph TD
    subgraph Entry & Configuration
        A[Input Text] --> MP[master_processor.py];
        MC[master_config.py] --> MP;
        GC[config/graph_config.py] --> MP;
    end

    subgraph Core Processing in master_processor.py
        MP --"1. _hierarchical_partition"--> P1;
        P1(Partitioning Logic) --"uses"--> AE[models/attention_extractor.py];
        P1 --> Segments;
        
        Segments --> P2("2. Graph Building");
        P2 --"uses"--> AGB[graph/attention_graph_builder.py];
        AGB --"uses"--> ABED[graph/attention_based_edge_detector.py];
        AGB --"uses"--> EED[graph/enhanced_edge_detector.py];
        
        P2 --> InitialGraph;
        
        InitialGraph --> P3("3. Graph Management");
        P3 --"uses"--> KGM[graph/knowledge_graph_manager.py];
        KGM --> CondensedGraph;
        
        CondensedGraph --> P4("4. Hierarchy Building");
        P4 --"uses"--> HGB[graph/hierarchical_graph_builder.py];
        HGB --> HierarchicalGraph;
        
        HierarchicalGraph --> P5("5. Reassembly");
        P5 --"uses"--> GR[graph/graph_reassembler.py];
        GR --> ReassembledText;
    end

    subgraph GPU & Advanced Features
        MP --"can use"--> TSP[tests/torch_spectral_processor.py];
        TSP --"uses"--> TAGB[graph/torch_attention_graph_builder.py];
        TAGB --"uses"--> TSC[graph/torch_spectral_clustering.py];
    end
    
    subgraph Output
        ReassembledText --> FinalOutput[Final Output Document];
    end

    style MP fill:#f9f,stroke:#333,stroke-width:2px
