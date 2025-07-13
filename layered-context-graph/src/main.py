try:
    # Try relative imports first (when used as a module)
    from .models.attention_extractor import AttentionExtractor, EnhancedAttentionExtractor
    from .models.context_window import ContextWindow
    from .models.knowledge_graph import KnowledgeGraph
    from .partitioning.instruction_seeder import InstructionSeeder
    from .partitioning.partition_manager import PartitionManager
    from .partitioning.percolation_analyzer import PercolationAnalyzer
    from .graph.graph_builder import GraphBuilder
    from .graph.node_classifier import NodeClassifier
    from .graph.graph_reassembler import GraphReassembler
    from .utils.notebook_parser import NotebookParser
    from .utils.visualization import Visualization
    from .ollama_integration import LayeredContextGraphIntegration
    from .config import OLLAMA_CONFIG, DEFAULT_CONFIG
except ImportError:
    # Fall back to absolute imports (when run directly)
    from models.attention_extractor import AttentionExtractor, EnhancedAttentionExtractor
    from models.context_window import ContextWindow
    from models.knowledge_graph import KnowledgeGraph
    from partitioning.instruction_seeder import InstructionSeeder
    from partitioning.partition_manager import PartitionManager
    from partitioning.percolation_analyzer import PercolationAnalyzer
    from graph.graph_builder import GraphBuilder
    from graph.node_classifier import NodeClassifier
    from graph.graph_reassembler import GraphReassembler
    from utils.notebook_parser import NotebookParser
    from utils.visualization import Visualization
    from ollama_integration import LayeredContextGraphIntegration
    from config import OLLAMA_CONFIG, DEFAULT_CONFIG


class LayeredContextGraph:
    """Main class for Layered Context Graph processing with support for multiple model types"""
    
    def __init__(self, config=None, model_type="transformer", model_name=None):
        self.config = config or {}
        self.model_type = model_type
        self.model_name = model_name
        
        # Load default config for model type
        from config import DEFAULT_CONFIG
        if model_type in DEFAULT_CONFIG:
            default_config = DEFAULT_CONFIG[model_type].copy()
            default_config.update(self.config)
            self.config = default_config
        
        # Initialize appropriate attention extractor
        if model_type == "ollama" and model_name:
            self.attention_extractor = EnhancedAttentionExtractor(
                model_source=model_name, 
                model_type="ollama"
            )
        else:
            # Use backward-compatible AttentionExtractor for transformer models
            self.attention_extractor = AttentionExtractor()
            
        # Initialize other components
        # Get window size from config or use default
        window_size = self.config.get('window_size', 8192)
        self.context_window = ContextWindow(window_size)
        self.knowledge_graph = KnowledgeGraph()
        # Remove unused partitioning components that were causing duplication
        self.graph_builder = GraphBuilder()
        self.node_classifier = NodeClassifier()
        self.graph_reassembler = GraphReassembler()
        self.notebook_parser = NotebookParser()
        self.visualization = Visualization()
        
    def process(self, text_or_notebook):
        """Process text or notebook through the Layered Context Graph pipeline"""
        
        print(f"🚀 Processing with {self.model_type} model")
        
        # Handle different input types
        if isinstance(text_or_notebook, str):
            if text_or_notebook.endswith('.ipynb'):
                # Parse notebook
                notebook = self.notebook_parser.parse_notebook(text_or_notebook)
                windows = self.context_window.create_window(notebook)
            else:
                # Direct text input - single semantic partitioning step
                windows = self.context_window.create_window(text_or_notebook)
        else:
            windows = self.context_window.create_window(text_or_notebook)
        
        print(f"📝 Created {len(windows)} semantic windows")
        
        # Let the transformer CREATE the graph directly from semantic windows
        # No pre-processing, no redundant partitioning - just transform and build
        for i, window in enumerate(windows):
            node_id = f"semantic_chunk_{i}"
            
            # Let the attention extractor analyze the window content
            # It should NOT modify the text, just extract patterns
            attention_data = self.attention_extractor.extract_attention(window)
            
            # Store ORIGINAL content as the primary data
            node_attributes = {
                'content': window,  # Original semantic chunk
                'original_text': window,  # Backup
                'attention_patterns': attention_data,  # Analysis metadata only
                'model_type': self.model_type,
                'model_name': getattr(self, 'model_name', 'unknown'),
                'chunk_index': i
            }
            
            # Add node to knowledge graph 
            self.knowledge_graph.add_node(node_id, node_attributes)
            print(f"  ✅ Added node {node_id}: {len(window)} chars")
        
        # Build edges between related semantic chunks
        nodes_list = list(self.knowledge_graph.nodes.keys())
        edges_list = self.knowledge_graph.edges
        self.graph_builder.build_graph(nodes_list, edges_list)
        
        # Classify nodes by content type
        for node_id in self.knowledge_graph.nodes:
            self.node_classifier.classify_node(node_id)
        
        print(f"🔗 Built graph with {len(nodes_list)} nodes and {len(edges_list)} edges")
        
        # Convert to reassembler format and apply organization rules
        nodes_for_reassembler = []
        for node_id, attributes in self.knowledge_graph.nodes.items():
            content = attributes.get('original_text', '') if isinstance(attributes, dict) else str(attributes)
            
            node_data = {
                'id': node_id,
                'content': content,
                'importance': 1.0,
                'attributes': attributes
            }
            nodes_for_reassembler.append(node_data)
        
        edges_for_reassembler = []
        for edge in self.knowledge_graph.edges:
            if len(edge) >= 2:
                edge_data = {
                    'source': edge[0],
                    'target': edge[1],
                    'weight': edge[2] if len(edge) > 2 else 1.0
                }
                edges_for_reassembler.append(edge_data)
        
        # Apply final organization using ORIGINAL TEXT as reconstruction seed
        reassembled_result = self.graph_reassembler.reassemble_graph(
            nodes_for_reassembler, 
            edges_for_reassembler,
            original_document=text_or_notebook  # Use original as reconstruction seed!
        )
        
        if isinstance(reassembled_result, dict):
            reassembled_result['original_edges'] = edges_for_reassembler
            reassembled_result['original_document'] = text_or_notebook
        
        return reassembled_result
    
    def visualize(self, graph=None):
        """Visualize the knowledge graph"""
        if graph is None:
            graph = self.knowledge_graph
        self.visualization.plot_graph(graph)


def main():
    # Example usage with different model types
    
    # Traditional transformer model
    # lcg_transformer = LayeredContextGraph(model_type="transformer")
    
    # Ollama model
    # lcg_ollama = LayeredContextGraph(model_type="ollama", model_name="qwq32b")
    
    # Original workflow for backward compatibility
    # Initialize components
    attention_extractor = AttentionExtractor()
    context_window = ContextWindow()
    knowledge_graph = KnowledgeGraph()
    instruction_seeder = InstructionSeeder()
    partition_manager = PartitionManager()
    percolation_analyzer = PercolationAnalyzer()
    graph_builder = GraphBuilder()
    node_classifier = NodeClassifier()
    graph_reassembler = GraphReassembler()
    notebook_parser = NotebookParser()
    visualization = Visualization()

    # Example workflow
    # 1. Parse a notebook
    notebook = notebook_parser.parse_notebook("path/to/notebook.ipynb")
    
    # 2. Extract context windows
    windows = context_window.create_window(notebook)
    
    # 3. Seed instructions
    seeded_text = instruction_seeder.seed_instructions(windows)
    
    # 4. Analyze partitions
    partitions = partition_manager.create_partitions(seeded_text)
    percolation_analyzer.analyze_percolation(partitions)
    
    # 5. Build knowledge graph
    for partition in partitions:
        attention_data = attention_extractor.extract_attention(partition)
        knowledge_graph.add_node(attention_data)
    
    # 6. Update graph
    graph_builder.build_graph(knowledge_graph)
    
    # 7. Classify nodes
    for node in knowledge_graph.nodes:
        node_classifier.classify_node(node)
    
    # 8. Reassemble graph
    optimized_graph = graph_reassembler.reassemble_graph(knowledge_graph)
    
    # 9. Visualize the graph
    visualization.plot_graph(optimized_graph)

if __name__ == "__main__":
    main()