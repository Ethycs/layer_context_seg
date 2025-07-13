"""
Enhanced Attention Extractor that supports both Transformer and Ollama GGUF models
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging

try:
    # Try relative imports first (when used as a module)
    from .ollama_extractor import OllamaModelExtractor
except ImportError:
    # Fall back to absolute imports (when run directly)
    from ollama_extractor import OllamaModelExtractor

logger = logging.getLogger(__name__)


class EnhancedAttentionExtractor:
    """
    Unified attention extractor supporting multiple model types:
    - Hugging Face Transformer models (runtime attention)
    - Ollama GGUF models (static weight analysis)
    """
    
    def __init__(self, model_source: Union[str, object], model_type: str = "auto"):
        """
        Initialize the attention extractor
        
        Args:
            model_source: Either a model name (str) or model object
            model_type: "transformer", "ollama", or "auto" (auto-detect)
        """
        self.model_source = model_source
        self.model_type = model_type
        self.model = None
        self.ollama_extractor = None
        self.ollama_config = None
        self.attention_cache = {}
        
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the appropriate model based on type"""
        
        if self.model_type == "auto":
            self._detect_model_type()
            
        if self.model_type == "ollama":
            self._initialize_ollama_model()
        elif self.model_type == "transformer":
            self._initialize_transformer_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
    def _detect_model_type(self):
        """Auto-detect model type based on source"""
        
        if isinstance(self.model_source, str):
            # Check if it's an Ollama model name
            ollama_models = ["qwq32b", "llama", "mistral", "qwen"]
            if any(model in self.model_source.lower() for model in ollama_models):
                self.model_type = "ollama"
            else:
                self.model_type = "transformer"
        else:
            # Assume it's a transformer model object
            self.model_type = "transformer"
            
    def _initialize_ollama_model(self):
        """Initialize Ollama GGUF model for static weight analysis"""
        
        logger.info(f"Initializing Ollama model: {self.model_source}")
        
        # Resolve model name to file path
        model_path = self._resolve_ollama_model_path(self.model_source)
        
        # Use the new QwQ 32B extractor
        try:
            logger.info(f"Loading GGUF model from: {model_path}")
            extractor = OllamaModelExtractor(model_path)
            
            # Get model configuration
            config = extractor.get_model_config()
            logger.info(f"GGUF model loaded successfully: {config}")
            
            # Extract attention weights for first layer as example
            attention_weights = extractor.extract_attention_weights(layer_idx=0)
            
            # Handle dictionary of weights vs single tensor
            if isinstance(attention_weights, dict):
                logger.info(f"Attention weights extracted: {len(attention_weights)} tensors")
                for key, tensor in attention_weights.items():
                    logger.info(f"  {key}: {tensor.shape if hasattr(tensor, 'shape') else type(tensor)}")
                    
                if torch.cuda.is_available():
                    # Move all tensors to GPU
                    for key in attention_weights:
                        if hasattr(attention_weights[key], 'to'):
                            attention_weights[key] = attention_weights[key].to("cuda")
            else:
                logger.info(f"Attention weights shape: {attention_weights.shape}")
                
                if torch.cuda.is_available():
                    logger.info("CUDA available, transferring attention weights to GPU")
                    attention_weights = attention_weights.to("cuda")
        except Exception as e:
            logger.error(f"Error initializing Ollama model: {e}")
            raise
        
        # Store for analysis
        self.ollama_extractor = extractor
        self.ollama_config = config
        self.ollama_attention_weights = attention_weights
        
    def _resolve_ollama_model_path(self, model_name: str) -> str:
        """Resolve model name to actual GGUF file path"""
        
        import os
        
        # Handle full paths directly
        if os.path.exists(model_name):
            return model_name
            
        # Try common model name mappings
        model_mappings = {
            'qwq': ['qwq.gguf'],
            'qwq32b': ['qwq.gguf'],
            'qwq-32b': ['qwq.gguf']
        }
        
        # Get possible filenames for this model
        possible_files = model_mappings.get(model_name.lower(), [f"{model_name}.gguf"])
        
        # Search locations
        search_paths = [
            "/workspace",
            "/workspace/layered-context-graph",
            ".",
            "./models",
            "../models"
        ]
        
        # Try to find the file
        for base_path in search_paths:
            for filename in possible_files:
                full_path = os.path.join(base_path, filename)
                if os.path.exists(full_path):
                    logger.info(f"✅ Found model file: {full_path}")
                    return full_path
        
        # If not found, raise informative error
        searched_locations = []
        for base_path in search_paths:
            for filename in possible_files:
                searched_locations.append(os.path.join(base_path, filename))
        
        raise FileNotFoundError(
            f"Could not find GGUF model file for '{model_name}'. "
            f"Searched locations: {searched_locations}"
        )
        
    def _initialize_transformer_model(self):
        """Initialize Hugging Face transformer model"""
        
        if isinstance(self.model_source, str):
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError:
                raise ImportError(
                    "transformers library is required for transformer models. "
                    "Install it with: pip install transformers>=4.20.0"
                )
            
            logger.info(f"Loading transformer model: {self.model_source}")
            self.model = AutoModel.from_pretrained(
                self.model_source,
                output_attentions=True,
                attn_implementation="eager"  # Fix for BERT attention warning
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_source)
        else:
            self.model = self.model_source
            
    def extract_attention(self, text_or_windows: Union[str, List[str]]) -> Dict:
        """
        Extract attention patterns from text or context windows
        
        Args:
            text_or_windows: Single text string or list of context windows
            
        Returns:
            Dictionary containing attention weights and patterns
        """
        
        if isinstance(text_or_windows, str):
            text_or_windows = [text_or_windows]
            
        if self.model_type == "ollama":
            return self._extract_ollama_attention(text_or_windows)
        else:
            return self._extract_transformer_attention(text_or_windows)
            
    def _extract_ollama_attention(self, windows: List[str]) -> Dict:
        """Extract attention using Ollama model static weights and PyTorch integration"""
        
        all_patterns = []
        
        # Check if extractor is properly initialized
        if not self.ollama_extractor:
            logger.error("Ollama extractor not initialized")
            return {
                'model_type': 'ollama',
                'model_name': self.model_source,
                'error': 'Extractor not initialized',
                'window_patterns': []
            }
        
        # For each window, analyze using PyTorch-enabled GGUF model
        for window_idx, window in enumerate(windows):
            logger.info(f"Processing window {window_idx+1}/{len(windows)} ({len(window)} chars)")
            
            try:
                # Get attention patterns for this window
                attention_patterns = self.ollama_extractor.get_attention_patterns(window)
                
                # Detect optimal segment boundaries based on attention
                boundary_scores = self.ollama_extractor.analyze_attention_for_boundaries(window)
                boundaries = self.ollama_extractor.detect_best_boundaries(window, num_segments=5)
                
                window_data = {
                    'window_idx': window_idx,
                    'text': window,
                    'attention_patterns': attention_patterns,
                    'boundary_scores': boundary_scores,
                    'suggested_boundaries': boundaries,
                    'model_info': self.ollama_config
                }
                
                all_patterns.append(window_data)
                logger.info(f"Window {window_idx} processed successfully")
                
            except Exception as e:
                logger.error(f"Error processing window {window_idx}: {e}")
                all_patterns.append({
                    'window_idx': window_idx,
                    'text': window,
                    'error': str(e)
                })
        
        return {
            'model_type': 'ollama',
            'model_name': self.model_source,
            'window_patterns': all_patterns,
            'model_config': self.ollama_config,
            'n_layers': self.ollama_config.get('n_layers', 32),
            'n_heads': self.ollama_config.get('n_heads', 32)
        }
        
    def _extract_transformer_attention(self, windows: List[str]) -> Dict:
        """Extract attention using transformer model runtime"""
        
        all_attentions = []
        
        with torch.no_grad():
            for window in windows:
                # Tokenize
                inputs = self.tokenizer(
                    window,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                
                # Forward pass
                outputs = self.model(**inputs)
                
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    # Stack all attention layers
                    attention_stack = torch.stack(outputs.attentions)
                    all_attentions.append(attention_stack)
                    
        return {
            'model_type': 'transformer',
            'model_name': self.model_source if isinstance(self.model_source, str) else "custom",
            'attention_tensors': all_attentions,
            'n_layers': len(all_attentions[0]) if all_attentions else 0,
            'n_heads': all_attentions[0].shape[2] if all_attentions else 0
        }
        
    def analyze_patterns(self, attention_data: Dict) -> Dict:
        """
        Analyze attention patterns to identify semantic relationships
        
        Args:
            attention_data: Output from extract_attention
            
        Returns:
            Analysis results including clusters and relationships
        """
        
        analysis = {
            'pattern_clusters': [],
            'semantic_groups': [],
            'attention_flow': []
        }
        
        if attention_data['model_type'] == 'ollama':
            analysis.update(self._analyze_ollama_patterns(attention_data))
        else:
            analysis.update(self._analyze_transformer_patterns(attention_data))
            
        return analysis
        
    def _analyze_ollama_patterns(self, attention_data: Dict) -> Dict:
        """Analyze patterns from Ollama model loaded in PyTorch"""
        
        analysis_results = {
            'segment_boundaries': [],
            'layer_patterns': {},
            'head_specializations': {}
        }
        
        # Process each window
        for window_data in attention_data['window_patterns']:
            window_idx = window_data.get('window_idx')
            
            # Skip windows with errors
            if 'error' in window_data:
                continue
            
            # Process boundaries
            boundaries = window_data.get('suggested_boundaries', [])
            if boundaries:
                analysis_results['segment_boundaries'].append({
                    'window_idx': window_idx,
                    'boundaries': boundaries,
                    'boundary_scores': window_data.get('boundary_scores', [])
                })
            
            # Process attention patterns
            attention_patterns = window_data.get('attention_patterns', {})
            
            # Analyze head specializations
            head_specializations = self._detect_head_specializations(attention_patterns)
            if head_specializations:
                analysis_results['head_specializations'][window_idx] = head_specializations
        
        return analysis_results
        
        # Identify layer clusters using simple thresholding
        clusters = []
        threshold = 0.75
        visited = set()
        
        for i in range(n_layers):
            if i in visited:
                continue
                
            cluster = [i]
            visited.add(i)
            
            for j in range(n_layers):
                if j not in visited and similarity_matrix[i, j] > threshold:
                    cluster.append(j)
                    visited.add(j)
                    
            if len(cluster) > 1:
                clusters.append(cluster)
                
        return {
            'layer_clusters': clusters,
            'similarity_matrix': similarity_matrix.tolist(),
            'isolated_layers': [i for i in range(n_layers) if i not in visited]
        }
        
    def _analyze_transformer_patterns(self, attention_data: Dict) -> Dict:
        """Analyze patterns from transformer model"""
        
        attention_tensors = attention_data['attention_tensors']
        
        if not attention_tensors:
            return {}
            
        # Compute attention statistics
        patterns = []
        
        for window_attention in attention_tensors:
            # Average attention across heads for each layer
            layer_patterns = window_attention.mean(dim=2)  # Average over heads
            patterns.append(layer_patterns)
            
        # Identify high-attention regions
        high_attention_threshold = 0.7
        high_attention_regions = []
        
        for idx, pattern in enumerate(patterns):
            regions = torch.where(pattern > high_attention_threshold)
            high_attention_regions.append({
                'window': idx,
                'regions': [(int(r[0]), int(r[1]), int(r[2])) for r in zip(*regions)]
            })
            
        return {
            'attention_statistics': {
                'mean': [p.mean().item() for p in patterns],
                'std': [p.std().item() for p in patterns],
                'max': [p.max().item() for p in patterns]
            },
            'high_attention_regions': high_attention_regions
        }
        
    def create_attention_graph(self, attention_data: Dict, text_chunks: List[str]) -> Dict:
        """
        Create a graph representation based on attention patterns
        
        Args:
            attention_data: Output from extract_attention
            text_chunks: Text chunks corresponding to nodes
            
        Returns:
            Graph structure with nodes and edges
        """
        
        nodes = []
        edges = []
        
        # Create nodes from text chunks
        for idx, chunk in enumerate(text_chunks):
            node = {
                'id': idx,
                'text': chunk,
                'attention_features': {},
                'word_count': len(chunk.split()),
                'char_count': len(chunk)
            }
            
            # Add attention-based features based on model type
            if attention_data['model_type'] == 'transformer' and attention_data.get('attention_tensors'):
                # For transformer models, use attention tensor features
                if idx < len(attention_data['attention_tensors']):
                    attention_tensor = attention_data['attention_tensors'][idx]
                    # Create a simple feature vector from attention
                    if len(attention_tensor) > 0:
                        # Average across layers and heads for a simple fingerprint
                        feature_vector = attention_tensor.mean(dim=(0, 1, 2))  # Average all dimensions
                        node['attention_features']['fingerprint'] = feature_vector
                        
            elif attention_data['model_type'] == 'ollama':
                # For Ollama models, create features from text characteristics
                # This is a fallback when attention patterns aren't available
                words = chunk.split()
                feature_vector = torch.tensor([
                    len(words),  # word count
                    len(chunk),  # character count
                    chunk.count('.'),  # sentence count approximation
                    chunk.count(','),  # complexity indicator
                    sum(1 for w in words if len(w) > 6),  # complex words
                ], dtype=torch.float32)
                node['attention_features']['fingerprint'] = feature_vector
                        
            nodes.append(node)
            
        # Create edges based on similarity
        similarity_threshold = 0.3  # Lower threshold for more connections
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                similarity = 0.0
                
                # Try attention-based similarity first
                if ('fingerprint' in nodes[i]['attention_features'] and 
                    'fingerprint' in nodes[j]['attention_features']):
                    try:
                        fingerprint_i = nodes[i]['attention_features']['fingerprint']
                        fingerprint_j = nodes[j]['attention_features']['fingerprint']
                        
                        # Ensure both are tensors and same size
                        if (isinstance(fingerprint_i, torch.Tensor) and 
                            isinstance(fingerprint_j, torch.Tensor) and
                            fingerprint_i.shape == fingerprint_j.shape):
                            
                            similarity = torch.nn.functional.cosine_similarity(
                                fingerprint_i.unsqueeze(0),
                                fingerprint_j.unsqueeze(0)
                            ).item()
                    except Exception as e:
                        # Fall back to text-based similarity
                        similarity = self._text_similarity(nodes[i]['text'], nodes[j]['text'])
                else:
                    # Use text-based similarity as fallback
                    similarity = self._text_similarity(nodes[i]['text'], nodes[j]['text'])
                
                # Create edge if similarity is above threshold
                if similarity > similarity_threshold:
                    edges.append({
                        'source': i,
                        'target': j,
                        'weight': similarity,
                        'type': 'semantic_similarity'
                    })
                        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'model_type': attention_data['model_type'],
                'n_nodes': len(nodes),
                'n_edges': len(edges),
                'similarity_threshold': similarity_threshold
            }
        }
    
    def _detect_head_specializations(self, attention_patterns: Dict) -> Dict:
        """
        Detect what each attention head specializes in, similar to the pattern described
        in the condensed architecture document.
        
        Args:
            attention_patterns: Dictionary of layer -> attention tensor
            
        Returns:
            Dictionary of head specializations
        """
        specializations = {
            'boundary_heads': [],
            'relation_heads': [],
            'cluster_heads': []
        }
        
        if not attention_patterns:
            return specializations
            
        # Calculate scores for each head across layers
        for layer_idx, layer_attention in attention_patterns.items():
            # Skip if not a tensor
            if not isinstance(layer_attention, torch.Tensor):
                continue
                
            n_heads = layer_attention.shape[0] if layer_attention.dim() > 2 else 1
            
            for head_idx in range(n_heads):
                # Extract this head's attention
                if layer_attention.dim() > 2:
                    head_attention = layer_attention[head_idx]
                else:
                    head_attention = layer_attention
                    
                # Calculate boundary detection score
                # High score = attention focuses within segments rather than across
                diagonal_attention = torch.diagonal(head_attention)
                off_diagonal = head_attention - torch.diag(diagonal_attention)
                boundary_score = diagonal_attention.mean().item() / (off_diagonal.mean().item() + 1e-6)
                
                # Calculate relation detection score
                # High score = attention connects related but distant tokens
                seq_len = head_attention.shape[0]
                if seq_len > 10:  # Need enough context to measure
                    distant_attn = torch.triu(head_attention, diagonal=seq_len//2)
                    relation_score = distant_attn.sum().item() / (head_attention.sum().item() + 1e-6)
                else:
                    relation_score = 0
                
                # Calculate clustering ability score
                # High score = attention forms clear clusters
                cluster_score = 0
                if seq_len > 5:
                    # Simple heuristic: variance of row-wise attention
                    row_vars = torch.var(head_attention, dim=1)
                    cluster_score = row_vars.mean().item()
                
                # Record this head's specializations
                head_key = (int(layer_idx), int(head_idx))
                
                # Add to appropriate specialization list if score is high enough
                if boundary_score > 1.5:  # Threshold determined empirically
                    specializations['boundary_heads'].append((head_key, boundary_score))
                
                if relation_score > 0.2:  # Threshold determined empirically
                    specializations['relation_heads'].append((head_key, relation_score))
                
                if cluster_score > 0.1:  # Threshold determined empirically
                    specializations['cluster_heads'].append((head_key, cluster_score))
        
        # Sort by score and take top-k
        k = 5  # Take top 5 heads for each specialization
        for spec_type in specializations:
            specializations[spec_type] = sorted(
                specializations[spec_type], 
                key=lambda x: x[1], 
                reverse=True
            )[:k]
            
        return specializations
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using simple word overlap
        
        Args:
            text1, text2: Text strings to compare
            
        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


# Convenience functions for integration
def create_attention_extractor(model_name: str = "qwq32b") -> EnhancedAttentionExtractor:
    """Create an attention extractor for the specified model"""
    return EnhancedAttentionExtractor(model_name)


class AttentionExtractor:
    """Backward compatibility wrapper for the original AttentionExtractor interface"""
    
    def __init__(self, model=None):
        """Initialize with optional model parameter for backward compatibility"""
        if model is None:
            # Use default transformer model
            model_name = "bert-base-uncased"
            self.extractor = EnhancedAttentionExtractor(
                model_source=model_name,
                model_type="transformer"
            )
        else:
            # Try to determine model type and name from the model object
            if hasattr(model, 'name_or_path'):
                model_name = model.name_or_path
            else:
                model_name = "bert-base-uncased"
            
            self.extractor = EnhancedAttentionExtractor(
                model_source=model_name,
                model_type="transformer"
            )
            # Store the original model for direct access if needed
            self.model = model
    
    def extract_attention(self, context_windows):
        """Extract attention patterns from context windows"""
        return self.extractor.extract_attention(context_windows)
    
    def process_attention(self, attention_weights):
        """Process attention weights (backward compatibility method)"""
        # Simple processing for backward compatibility
        processed_attention = []
        for weights in attention_weights:
            if hasattr(weights, 'mean'):
                processed_attention.append(weights.mean(dim=1))
            else:
                processed_attention.append(weights)
        return processed_attention