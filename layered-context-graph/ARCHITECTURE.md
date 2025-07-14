# Layered Context Graph System - Architecture Guide v2.0

## Core Philosophy

The Layered Context Graph System follows a **Source-First, Scaffold-Guided** approach with **Multi-Round Annotation** capability where:
- **Fluff removal happens at source** during initial partitioning
- **Transformers create graphs directly** from semantic chunks
- **Multi-round analysis** enriches same graph with layered annotations
- **Original document serves as reconstruction scaffold** for coherent reassembly
- **Configurable processing** supports both single-pass and multi-round modes

## Revolutionary Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Text    │───▶│  Semantic Windows │───▶│ Base Graph      │
│  (Original Doc) │    │  + Fluff Removal │    │ Construction    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                                               │
        │ (serves as scaffold)                          ▼
        │                        ┌─────────────────────────────────┐
        │                        │     Multi-Round Annotation     │
        │                        │  ┌─────┐ ┌─────┐ ┌─────┐      │
        │                        │  │Round│ │Round│ │Round│      │
        │                        │  │  1  │ │  2  │ │  3  │      │
        │                        │  └─────┘ └─────┘ └─────┘      │
        │                        │     ↓       ↓       ↓         │
        │                        │  Layer   Layer   Layer        │
        │                        │    1       2       3          │
        │                        └─────────────────────────────────┘
        │                                        │
        └──────────────────────────────────────▶ │
                                                 ▼
                                      ┌─────────────────┐
                                      │ Scaffold-Guided │
                                      │ Reconstruction  │
                                      └─────────────────┘
```

## Key Innovations

### 1. Single Graph, Multiple Analysis Layers
- **Base Graph**: Created once from clean semantic chunks
- **Annotation Layers**: Multiple analysis rounds add metadata to same nodes/edges
- **Layer Types**: Different semantic perspectives (syntactic, semantic, pragmatic)
- **Accumulative**: Each round enriches existing graph structure

### 2. Fluff Removal at Source
- **Where**: During semantic window creation in `ContextWindow._create_semantic_windows()`
- **What**: Eliminates filler words, duplicate sentences, repetitive phrases
- **Why**: Prevents transformers from processing redundant content
- **Result**: Cleaner graphs, no post-processing deduplication needed

### 3. Transformer-Driven Graph Creation
- **Principle**: Attention mechanisms slice existing text, never recreate it
- **Process**: Transformer analyzes semantic chunks and builds knowledge graph
- **Multi-Round Capability**: Same graph enriched through multiple analysis passes
- **Output**: Pure graph structure with original content preserved

### 4. Scaffold-Guided Reconstruction
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

#### Multi-Round Annotation System (New)
- **Purpose**: Enriches base graph through multiple analysis layers
- **Key Innovation**: Single graph structure with accumulative annotations
- **Analysis Rounds**:
  1. **Syntactic Layer**: Grammar, POS tags, linguistic structure
  2. **Semantic Layer**: Topics, concepts, meaning relationships
  3. **Pragmatic Layer**: Intent, discourse, communicative purpose
- **Process**: Each round adds layer-specific metadata to existing nodes/edges
- **Cross-Layer Synthesis**: Combines insights from all layers

#### AnnotationLayer (Core Component)
```python
class AnnotationLayer:
    def __init__(self, layer_type, analyzer):
        self.layer_type = layer_type  # 'syntactic', 'semantic', 'pragmatic'
        self.analyzer = analyzer      # Different transformer models/approaches
        self.metadata_prefix = f"layer_{layer_type}_"
    
    def annotate_graph(self, graph):
        """Add layer-specific annotations to existing graph"""
        for node in graph.nodes():
            node_annotations = self.analyzer.analyze_node(node.content)
            for key, value in node_annotations.items():
                node.metadata[f"{self.metadata_prefix}{key}"] = value
        
        for edge in graph.edges():
            edge_annotations = self.analyzer.analyze_edge(edge.source, edge.target)
            for key, value in edge_annotations.items():
                edge.metadata[f"{self.metadata_prefix}{key}"] = value
```

#### Analysis Layer Types

##### Round 1: Syntactic Analyzer
```python
class SyntacticAnalyzer:
    """Focuses on grammatical structure and linguistic patterns"""
    
    def analyze_node(self, content):
        return {
            'pos_tags': self.extract_pos_patterns(content),
            'syntax_complexity': self.calculate_complexity(content),
            'sentence_structures': self.identify_structures(content),
            'linguistic_features': self.extract_features(content)
        }
    
    def analyze_edge(self, source_node, target_node):
        return {
            'syntactic_similarity': self.calculate_syntax_similarity(source_node, target_node),
            'transition_type': self.identify_transition(source_node, target_node),
            'grammatical_coherence': self.measure_coherence(source_node, target_node)
        }
```

##### Round 2: Semantic Analyzer
```python
class SemanticAnalyzer:
    """Focuses on meaning and conceptual relationships"""
    
    def analyze_node(self, content):
        return {
            'semantic_topics': self.extract_topics(content),
            'concept_density': self.calculate_concept_density(content),
            'semantic_role': self.identify_semantic_role(content),
            'domain_classification': self.classify_domain(content)
        }
    
    def analyze_edge(self, source_node, target_node):
        return {
            'semantic_relation_type': self.classify_relation(source_node, target_node),
            'conceptual_distance': self.calculate_distance(source_node, target_node),
            'logical_flow': self.analyze_logical_connection(source_node, target_node)
        }
```

##### Round 3: Pragmatic Analyzer
```python
class PragmaticAnalyzer:
    """Focuses on context, intent, and communicative purpose"""
    
    def analyze_node(self, content):
        return {
            'communicative_intent': self.identify_intent(content),
            'discourse_function': self.classify_function(content),
            'contextual_importance': self.rate_importance(content),
            'rhetorical_device': self.identify_rhetoric(content)
        }
    
    def analyze_edge(self, source_node, target_node):
        return {
            'discourse_relation': self.identify_discourse_relation(source_node, target_node),
            'argumentative_flow': self.analyze_argument_flow(source_node, target_node),
            'contextual_relevance': self.measure_relevance(source_node, target_node)
        }
```

### 2. Graph Construction Layer (`src/graph/`)

#### AnnotatedKnowledgeGraph (Multi-Round Support)
```python
class AnnotatedKnowledgeGraph:
    """Knowledge graph with multi-layer annotation support"""
    
    def __init__(self):
        self.base_graph = None
        self.annotation_layers = {}
        self.layer_order = []
    
    def create_base_graph(self, semantic_chunks):
        """Create the foundational graph structure"""
        self.base_graph = self._build_graph_from_chunks(semantic_chunks)
        return self.base_graph
    
    def add_annotation_layer(self, layer_name, analyzer):
        """Add a new analysis layer to the graph"""
        if self.base_graph is None:
            raise ValueError("Base graph must be created first")
        
        layer = AnnotationLayer(layer_name, analyzer)
        layer.annotate_graph(self.base_graph)
        
        self.annotation_layers[layer_name] = layer
        self.layer_order.append(layer_name)
        
        return layer
    
    def get_node_annotations(self, node_id, layer=None):
        """Retrieve annotations for a specific node"""
        node = self.base_graph.nodes[node_id]
        
        if layer:
            return {k: v for k, v in node.metadata.items() 
                   if k.startswith(f"layer_{layer}_")}
        return node.metadata
    
    def get_layered_insights(self, node_id):
        """Get insights from all layers for a node"""
        insights = {}
        for layer_name in self.layer_order:
            layer_data = self.get_node_annotations(node_id, layer_name)
            insights[layer_name] = layer_data
        return insights
```

#### MultiRoundProcessor (Orchestrator)
```python
class MultiRoundProcessor:
    """Orchestrates multiple rounds of semantic analysis"""
    
    def __init__(self, analysis_config):
        self.analysis_rounds = self._setup_analysis_rounds(analysis_config)
        self.graph = AnnotatedKnowledgeGraph()
    
    def process_with_multi_round_analysis(self, semantic_chunks):
        """Process chunks through multiple analysis rounds"""
        
        # Step 1: Create base graph
        base_graph = self.graph.create_base_graph(semantic_chunks)
        
        # Step 2: Apply multiple analysis rounds
        for round_name, analyzer in self.analysis_rounds.items():
            print(f"Applying {round_name} analysis...")
            self.graph.add_annotation_layer(round_name, analyzer)
        
        # Step 3: Synthesize multi-layer insights
        enriched_graph = self._synthesize_layers()
        
        return enriched_graph
    
    def _synthesize_layers(self):
        """Combine insights from all layers into coherent annotations"""
        for node_id in self.graph.base_graph.nodes():
            layered_insights = self.graph.get_layered_insights(node_id)
            
            # Synthesize cross-layer patterns
            synthesis = self._cross_layer_synthesis(layered_insights)
            
            # Add synthesized insights to node
            node = self.graph.base_graph.nodes[node_id]
            node.metadata['synthesized_insights'] = synthesis
        
        return self.graph.base_graph
    
    def _cross_layer_synthesis(self, layered_insights):
        """Synthesize insights across different analysis layers"""
        synthesis = {
            'confidence_score': self._calculate_confidence(layered_insights),
            'semantic_density': self._calculate_semantic_density(layered_insights),
            'structural_importance': self._calculate_importance(layered_insights),
            'cross_layer_patterns': self._identify_patterns(layered_insights)
        }
        return synthesis
```

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

## Configuration (Multi-Round)

### Analysis Rounds Configuration
```python
MULTI_ROUND_CONFIG = {
    'analysis_rounds': {
        'syntactic': {
            'model': 'spacy_en_core_web_lg',
            'features': ['pos_tags', 'dependencies', 'syntax_patterns'],
            'weight': 0.2
        },
        'semantic': {
            'model': 'sentence-transformers/all-MiniLM-L6-v2',
            'features': ['topics', 'concepts', 'semantic_roles'],
            'weight': 0.5
        },
        'pragmatic': {
            'model': 'microsoft/DialoGPT-medium',
            'features': ['intent', 'discourse', 'rhetoric'],
            'weight': 0.3
        }
    },
    'synthesis': {
        'cross_layer_analysis': True,
        'confidence_threshold': 0.7,
        'layer_weight_normalization': True
    },
    'processing_mode': 'multi_round'  # or 'single_pass'
}
```

## Data Flow

### Single-Pass Mode (Default)
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

### Multi-Round Mode (Enhanced)
1. **Input Processing**: 
   ```
   Raw Text → Semantic Windowing → Fluff Removal → Clean Chunks
   ```

2. **Base Graph Construction**:
   ```
   Clean Chunks → Initial Attention Analysis → Base Graph Creation
   ```

3. **Multi-Round Annotation**:
   ```
   Base Graph → Round 1 (Syntactic) → Round 2 (Semantic) → Round 3 (Pragmatic) → Enriched Graph
   ```

4. **Scaffold-Guided Reconstruction**:
   ```
   Original Text + Enriched Graph → Multi-Layer Insights → Enhanced Reconstruction
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

## Benefits of Multi-Round Annotation

### 1. Rich Multi-Perspective Analysis
- **Comprehensive**: Multiple semantic perspectives on same content
- **Layered**: Different levels of analysis (syntax → semantics → pragmatics)
- **Efficient**: Single graph structure, multiple annotation passes

### 2. Preserved Graph Topology
- **Stable Structure**: Base graph topology remains unchanged
- **Additive Process**: Each round enriches without disrupting
- **Consistency**: Relationships maintained across all analysis layers

### 3. Enhanced Reconstruction Quality
- **Multi-Modal Insights**: Reconstruction informed by all analysis layers
- **Weighted Decisions**: Layer importance configurable for different domains
- **Cross-Layer Synthesis**: Patterns identified across analysis types

### 4. Flexible Processing Modes
- **Single-Pass**: Fast processing for simple use cases
- **Multi-Round**: Deep analysis for complex documents
- **Configurable**: Users can choose appropriate processing depth

## Usage Examples

### Multi-Round Processing
```python
from layered_context_graph import MultiRoundProcessor

# Configure multi-round analysis
config = MULTI_ROUND_CONFIG.copy()
processor = MultiRoundProcessor(config)

# Process document with multiple analysis rounds
semantic_chunks = create_semantic_windows(document)
enriched_graph = processor.process_with_multi_round_analysis(semantic_chunks)

# Access layered insights
for node_id in enriched_graph.nodes():
    syntactic_data = enriched_graph.get_node_annotations(node_id, 'syntactic')
    semantic_data = enriched_graph.get_node_annotations(node_id, 'semantic')
    pragmatic_data = enriched_graph.get_node_annotations(node_id, 'pragmatic')
    
    # Get synthesized insights
    synthesis = enriched_graph.nodes[node_id].metadata['synthesized_insights']
```

This architecture provides a robust foundation for transforming context windows into knowledge graphs while maintaining theoretical soundness and practical efficiency.