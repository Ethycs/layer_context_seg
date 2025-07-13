"""
Integration script for Ollama models with Layered Context Graph System
Connects extracted QWQ32B model attention patterns to graph construction
"""

import torch
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

try:
    # Try relative imports first (when used as a module)
    from .models.ollama_extractor import OllamaModelExtractor
    from .models.attention_extractor import EnhancedAttentionExtractor
except ImportError:
    # Fall back to absolute imports (when run directly)
    from models.ollama_extractor import OllamaModelExtractor
    from models.attention_extractor import EnhancedAttentionExtractor

logger = logging.getLogger(__name__)


class OllamaAttentionExtractor:
    """Modified AttentionExtractor that works with Ollama GGUF models"""
    
    def __init__(self, model_name: str = "qwq32b"):
        self.model_name = model_name
        self.extractor = None
        self.config = None
        
    def initialize(self):
        """Initialize the model and extract attention patterns"""
        
        logger.info(f"Initializing Ollama model: {self.model_name}")
        
        # Find the GGUF model file
        gguf_files = [f"/workspace/qwq.gguf", f"/workspace/layered-context-graph/qwq.gguf"]
        model_path = None
        for path in gguf_files:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            raise FileNotFoundError(f"No GGUF model file found for {self.model_name}")
        
        # Initialize the extractor with QwQ 32B support
        self.extractor = OllamaModelExtractor(model_path)
        self.config = self.extractor.get_model_config()
        
        logger.info(f"✅ Initialized QwQ 32B model with config: {self.config['model_type']}")
        
    def extract_attention(self, text: str, layer_indices: Optional[List[int]] = None) -> Dict:
        """
        Extract attention patterns for given text using QwQ 32B model weights
        
        Args:
            text: Input text to analyze
            layer_indices: Specific layers to analyze (None = first few layers)
            
        Returns:
            Dictionary containing attention weights and patterns
        """
        
        if not hasattr(self, 'extractor') or self.extractor is None:
            self.initialize()
            
        if layer_indices is None:
            # Use first 3 layers for demonstration
            layer_indices = [0, 1, 2]
            
        # Extract attention weights for specified layers
        attention_patterns = {
            'fingerprints': {},
            'layer_similarities': {},
            'head_analysis': {},
            'model_config': self.config
        }
        
        # Extract attention weights for requested layers
        for idx in layer_indices:
            try:
                # Extract attention weights for this layer
                layer_weights = self.extractor.extract_attention_weights(layer_idx=idx)
                if layer_weights:
                    attention_patterns['fingerprints'][idx] = layer_weights
                    logger.info(f"✅ Extracted attention weights for layer {idx}")
                else:
                    logger.warning(f"No attention weights found for layer {idx}")
            except Exception as e:
                logger.warning(f"Failed to extract weights for layer {idx}: {e}")
                
        # Analyze QwQ 32B specific properties
        attention_patterns['head_analysis'] = {
            'total_attention_heads': self.config['num_attention_heads'],
            'kv_heads': self.config['num_key_value_heads'],
            'grouped_query_attention': self.config.get('use_grouped_query_attention', False),
            'head_dimension': self.config['head_dim']
        }
        
        # Compute layer similarities based on attention patterns
        for i in range(len(layer_indices)):
            for j in range(i + 1, len(layer_indices)):
                layer1, layer2 = layer_indices[i], layer_indices[j]
                # Simple similarity based on available patterns
                if (layer1 in attention_patterns['fingerprints'] and 
                    layer2 in attention_patterns['fingerprints']):
                    similarity = 0.8  # Placeholder similarity score
                    attention_patterns['layer_similarities'][(layer1, layer2)] = similarity
                
        return attention_patterns
        
    def analyze_patterns(self, attention_weights: Dict) -> Dict:
        """Analyze attention patterns to identify semantic relationships"""
        
        analysis = {
            'cluster_layers': [],
            'unique_layers': [],
            'similarity_matrix': None
        }
        
        # Build similarity matrix
        layer_indices = list(attention_weights['fingerprints'].keys())
        n_layers = len(layer_indices)
        similarity_matrix = np.zeros((n_layers, n_layers))
        
        for i, layer1 in enumerate(layer_indices):
            for j, layer2 in enumerate(layer_indices):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                elif (layer1, layer2) in attention_weights['layer_similarities']:
                    similarity_matrix[i, j] = attention_weights['layer_similarities'][(layer1, layer2)]
                elif (layer2, layer1) in attention_weights['layer_similarities']:
                    similarity_matrix[i, j] = attention_weights['layer_similarities'][(layer2, layer1)]
                    
        analysis['similarity_matrix'] = similarity_matrix
        
        # Identify clusters of similar layers
        threshold = 0.8
        for i in range(n_layers):
            similar_layers = [layer_indices[j] for j in range(n_layers) 
                            if similarity_matrix[i, j] > threshold and i != j]
            if similar_layers:
                analysis['cluster_layers'].append({
                    'layer': layer_indices[i],
                    'similar_to': similar_layers
                })
            else:
                analysis['unique_layers'].append(layer_indices[i])
                
        return analysis


class LayeredContextGraphIntegration:
    """Integration class for Ollama models with Layered Context Graph"""
    
    def __init__(self, model_name: str = "qwq32b"):
        self.model_name = model_name
        self.attention_extractor = OllamaAttentionExtractor(model_name)
        self.attention_extractor.initialize()
        
    def create_attention_based_partitions(self, text: str, window_size: int = 8192) -> List[Dict]:
        """
        Create partitions based on Ollama model attention patterns
        
        Args:
            text: Input text to partition
            window_size: Size of context window
            
        Returns:
            List of partition dictionaries
        """
        
        # Extract attention patterns
        attention_data = self.attention_extractor.extract_attention(text)
        pattern_analysis = self.attention_extractor.analyze_patterns(attention_data)
        
        # Create partitions based on attention clusters
        partitions = []
        
        # Use unique layers for distinct partitions
        for layer_idx in pattern_analysis['unique_layers']:
            partition = {
                'layer': layer_idx,
                'type': 'unique',
                'fingerprint': attention_data['fingerprints'][layer_idx],
                'text_segments': []  # To be filled with actual text chunks
            }
            partitions.append(partition)
            
        # Use clustered layers for related content
        for cluster in pattern_analysis['cluster_layers']:
            partition = {
                'layer': cluster['layer'],
                'type': 'cluster',
                'similar_layers': cluster['similar_to'],
                'fingerprint': attention_data['fingerprints'][cluster['layer']],
                'text_segments': []
            }
            partitions.append(partition)
            
        return partitions
        
    def compute_edge_weights(self, node1_fingerprint: torch.Tensor, 
                           node2_fingerprint: torch.Tensor) -> float:
        """
        Compute edge weight between nodes using attention fingerprints
        
        Args:
            node1_fingerprint: Attention fingerprint of first node
            node2_fingerprint: Attention fingerprint of second node
            
        Returns:
            Edge weight (0-1)
        """
        
        # Compute cosine similarity between fingerprints
        similarity = torch.nn.functional.cosine_similarity(
            node1_fingerprint.unsqueeze(0),
            node2_fingerprint.unsqueeze(0)
        )
        
        return similarity.item()
        
    def create_knowledge_graph_config(self) -> Dict:
        """
        Create configuration for knowledge graph construction using Ollama model
        
        Returns:
            Configuration dictionary
        """
        
        config = {
            'model_name': self.model_name,
            'n_layers': self.attention_extractor.config.get('num_hidden_layers', 32),
            'n_heads': self.attention_extractor.config.get('num_attention_heads', 32),
            'hidden_size': self.attention_extractor.config.get('hidden_size', 4096),
            'attention_based_partitioning': True,
            'use_static_weights': True,  # Since we're using pre-extracted weights
            'percolation_threshold': 0.25,  # Maintain connectivity
            'instruction_markers': [
                '<QWQ_REASONING>',  # For QWQ model's reasoning patterns
                '<QWQ_CONCLUSION>',  # For conclusions
                '<QWQ_EXAMPLE>',     # For examples
                '<QWQ_DEFINITION>'   # For definitions
            ]
        }
        
        return config


def demonstrate_integration():
    """Demonstrate the integration of Ollama models with Layered Context Graph"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Initialize integration
    logger.info("Initializing Ollama-LCG integration...")
    integration = LayeredContextGraphIntegration("qwq32b")
    
    # Sample text for demonstration
    sample_text = """
    The layered context graph system transforms text into knowledge graphs using attention patterns.
    This allows for better information retention and retrieval across large context windows.
    By leveraging percolation theory, we ensure optimal connectivity between graph nodes.
    """
    
    # Create attention-based partitions
    logger.info("Creating attention-based partitions...")
    partitions = integration.create_attention_based_partitions(sample_text)
    
    logger.info(f"Created {len(partitions)} partitions:")
    for i, partition in enumerate(partitions):
        logger.info(f"  Partition {i}: Layer {partition['layer']} (Type: {partition['type']})")
        
    # Get configuration for knowledge graph
    config = integration.create_knowledge_graph_config()
    logger.info(f"Knowledge graph configuration: {config}")
    
    # Demonstrate edge weight calculation
    if len(partitions) >= 2:
        weight = integration.compute_edge_weights(
            partitions[0]['fingerprint'],
            partitions[1]['fingerprint']
        )
        logger.info(f"Edge weight between partition 0 and 1: {weight:.4f}")
        
    return integration, partitions, config


if __name__ == "__main__":
    # Run demonstration
    integration, partitions, config = demonstrate_integration()
    
    print("\nIntegration complete!")
    print(f"Model: {config['model_name']}")
    print(f"Layers: {config['n_layers']}")
    print(f"Attention heads: {config['n_heads']}")
    print(f"Hidden size: {config['hidden_size']}")