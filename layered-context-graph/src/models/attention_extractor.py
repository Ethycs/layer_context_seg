"""
QwQ-32B Attention Extractor - Primary model for layered context segmentation
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging

from models.ollama_extractor import OllamaModelExtractor

logger = logging.getLogger(__name__)


class EnhancedAttentionExtractor:
    """
    QwQ-32B attention extractor for layered context segmentation.
    This is the primary and only model used for tape splitting.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the QwQ attention extractor
        
        Args:
            model_path: Optional custom path to QwQ GGUF model file
        """
        self.model_path = model_path
        self.ollama_extractor = None
        self.ollama_config = None
        self.attention_cache = {}
        
        self._initialize_qwq_model()
        
    def _initialize_qwq_model(self):
        """Initialize QwQ-32B GGUF model for attention extraction"""
        
        logger.info("Initializing QwQ-32B model for attention extraction")
        
        # Resolve QwQ model path
        if self.model_path is None:
            self.model_path = self._find_qwq_model()
        
        try:
            logger.info(f"Loading QwQ GGUF model from: {self.model_path}")
            self.ollama_extractor = OllamaModelExtractor(self.model_path)
            
            # Get model configuration
            self.ollama_config = self.ollama_extractor.get_model_config()
            logger.info(f"QwQ model loaded successfully: {self.ollama_config}")
            
            # Extract attention weights for verification
            attention_weights = self.ollama_extractor.extract_attention_weights(layer_idx=0)
            
            # Handle attention weights and move to GPU if available
            if isinstance(attention_weights, dict):
                logger.info(f"QwQ attention weights extracted: {len(attention_weights)} tensors")
                for key, tensor in attention_weights.items():
                    logger.info(f"  {key}: {tensor.shape if hasattr(tensor, 'shape') else type(tensor)}")
                    
                if torch.cuda.is_available():
                    for key in attention_weights:
                        if hasattr(attention_weights[key], 'to'):
                            attention_weights[key] = attention_weights[key].to("cuda")
            else:
                logger.info(f"QwQ attention weights shape: {attention_weights.shape}")
                
                if torch.cuda.is_available():
                    logger.info("CUDA available, transferring QwQ attention weights to GPU")
                    attention_weights = attention_weights.to("cuda")
                    
        except Exception as e:
            logger.error(f"Error initializing QwQ model: {e}")
            raise RuntimeError(f"Failed to initialize QwQ-32B model: {e}")
        
        # Store attention weights for analysis
        self.qwq_attention_weights = attention_weights
        
    def _find_qwq_model(self) -> str:
        """Find QwQ GGUF model file in standard locations"""
        
        import os
        
        # QwQ model file patterns
        qwq_patterns = [
            'qwq.gguf',
            'qwq-32b.gguf', 
            'qwq32b.gguf',
            'QwQ-32B-Preview.gguf',
            'qwq-32b-preview.gguf'
        ]
        
        # Search locations
        search_paths = [
            "/workspace",
            "/workspaces/layer_context_seg",
            "/workspaces/layer_context_seg/layered-context-graph",
            ".",
            "./models",
            "../models",
            "~/models"
        ]
        
        # Expand home directory
        search_paths = [os.path.expanduser(path) for path in search_paths]
        
        # Find QwQ model file
        for base_path in search_paths:
            if not os.path.exists(base_path):
                continue
                
            for pattern in qwq_patterns:
                full_path = os.path.join(base_path, pattern)
                if os.path.exists(full_path):
                    logger.info(f"✅ Found QwQ model: {full_path}")
                    return full_path
        
        # If not found, provide helpful error
        searched_locations = []
        for base_path in search_paths:
            for pattern in qwq_patterns:
                searched_locations.append(os.path.join(base_path, pattern))
        
        raise FileNotFoundError(
            f"QwQ-32B model file not found. Please download QwQ GGUF model and place it in one of these locations:\n"
            f"Expected filenames: {qwq_patterns}\n"
            f"Searched locations: {searched_locations[:5]}... (and {len(searched_locations)-5} more)\n\n"
            f"Download from: https://huggingface.co/Qwen/QwQ-32B-Preview-GGUF"
        )
        
    def extract_attention_batch(self, windows: List[str]) -> List[Dict]:
        """
        Extract attention patterns for multiple windows in batch.
        More efficient than extract_attention_for_tape_splitting.
        
        Args:
            windows: List of context windows to analyze
            
        Returns:
            List of dictionaries with attention analysis for each window
        """
        logger.info(f"Batch processing {len(windows)} windows with QwQ-32B")
        
        # Use batch processing from ollama extractor
        if hasattr(self.ollama_extractor, 'get_attention_patterns_batch'):
            try:
                batch_results = self.ollama_extractor.get_attention_patterns_batch(windows)
                
                # Process results with head specialization detection
                processed_results = []
                for i, result in enumerate(batch_results):
                    if 'error' not in result:
                        head_specializations = self._detect_qwq_head_specializations(
                            result.get('attention_patterns', {})
                        )
                        result['head_specializations'] = head_specializations
                    processed_results.append(result)
                
                return processed_results
                
            except Exception as e:
                logger.error(f"Batch extraction failed: {e}")
                # Fall through to individual processing
        
        # Fallback to individual processing
        results = []
        for window in windows:
            attention_patterns = self.ollama_extractor.get_attention_patterns(window)
            boundary_scores = self.ollama_extractor.analyze_attention_for_boundaries(window)
            boundaries = self.ollama_extractor.detect_best_boundaries(window, num_segments=5)
            head_specializations = self._detect_qwq_head_specializations(attention_patterns)
            
            results.append({
                'attention_patterns': attention_patterns,
                'boundary_scores': boundary_scores,
                'optimal_boundaries': boundaries,
                'head_specializations': head_specializations
            })
        
        return results
    
    def extract_attention_for_tape_splitting(self, text_windows: List[str]) -> Dict:
        """
        Extract attention patterns from text windows using QwQ-32B model.
        This is the primary method for tape splitting.
        
        Args:
            text_windows: List of text segments to analyze
            
        Returns:
            Dictionary containing QwQ attention analysis for segmentation
        """
        
        logger.info(f"Extracting QwQ attention patterns for {len(text_windows)} windows")
        
        # Check cache first
        all_patterns = []
        uncached_windows = []
        uncached_indices = []
        
        for window_idx, window in enumerate(text_windows):
            cache_key = self._get_cache_key(window)
            if cache_key in self.attention_cache:
                logger.info(f"Using cached attention for window {window_idx}")
                cached_data = self.attention_cache[cache_key].copy()
                cached_data['window_idx'] = window_idx
                all_patterns.append(cached_data)
            else:
                uncached_windows.append(window)
                uncached_indices.append(window_idx)
        
        # Process uncached windows in batches
        if uncached_windows:
            logger.info(f"Processing {len(uncached_windows)} uncached windows in batches")
            
            # Use batch processing if available
            if hasattr(self.ollama_extractor, 'get_attention_patterns_batch'):
                try:
                    batch_results = self.ollama_extractor.get_attention_patterns_batch(uncached_windows)
                    
                    for i, (window_idx, window) in enumerate(zip(uncached_indices, uncached_windows)):
                        if i < len(batch_results):
                            batch_result = batch_results[i]
                            
                            # Detect specialized attention heads
                            head_specializations = self._detect_qwq_head_specializations(batch_result.get('attention_patterns', {}))
                            
                            window_data = {
                                'window_idx': window_idx,
                                'text': window,
                                'qwq_attention_patterns': batch_result.get('attention_patterns', {}),
                                'boundary_scores': batch_result.get('boundary_scores', {}),
                                'optimal_boundaries': batch_result.get('optimal_boundaries', []),
                                'head_specializations': head_specializations,
                                'model_info': self.ollama_config
                            }
                            
                            # Cache the result
                            cache_key = self._get_cache_key(window)
                            self.attention_cache[cache_key] = window_data.copy()
                            
                            all_patterns.append(window_data)
                            logger.info(f"Window {window_idx} processed with batch QwQ attention")
                        else:
                            logger.error(f"Batch processing failed for window {window_idx}")
                            all_patterns.append({
                                'window_idx': window_idx,
                                'text': window,
                                'error': 'Batch processing failed'
                            })
                
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}, falling back to individual processing")
                    # Fall back to individual processing
                    individual_results = self._process_windows_individually(uncached_windows, uncached_indices)
                    all_patterns.extend(individual_results)
            else:
                # Fall back to individual processing
                individual_results = self._process_windows_individually(uncached_windows, uncached_indices)
                all_patterns.extend(individual_results)
        
        # Sort patterns by window index to maintain order
        all_patterns.sort(key=lambda x: x['window_idx'])
        
        return {
            'model': 'qwq-32b',
            'segmentation_method': 'qwq_attention_heads',
            'window_patterns': all_patterns,
            'model_config': self.ollama_config,
            'n_layers': self.ollama_config.get('n_layers', 32),
            'n_heads': self.ollama_config.get('n_heads', 32),
            'attention_architecture': 'multi_head_attention'
        }
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for text content"""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _process_windows_individually(self, windows: List[str], indices: List[int]) -> List[Dict]:
        """Process windows individually when batch processing is not available"""
        results = []
        
        for window_idx, window in zip(indices, windows):
            logger.info(f"Processing window {window_idx+1}/{len(windows)} ({len(window)} chars)")
            
            try:
                # Validate text size before processing
                if len(window) > 50000:  # 50K char limit
                    logger.warning(f"Window {window_idx} too large ({len(window)} chars), chunking...")
                    chunked_results = self._process_large_window(window, window_idx)
                    results.extend(chunked_results)
                    continue
                
                # Get QwQ attention patterns for this window
                attention_patterns = self.ollama_extractor.get_attention_patterns(window)
                
                # Analyze attention for optimal segment boundaries
                boundary_scores = self.ollama_extractor.analyze_attention_for_boundaries(window)
                boundaries = self.ollama_extractor.detect_best_boundaries(window, num_segments=5)
                
                # Detect specialized attention heads
                head_specializations = self._detect_qwq_head_specializations(attention_patterns)
                
                window_data = {
                    'window_idx': window_idx,
                    'text': window,
                    'qwq_attention_patterns': attention_patterns,
                    'boundary_scores': boundary_scores,
                    'optimal_boundaries': boundaries,
                    'head_specializations': head_specializations,
                    'model_info': self.ollama_config
                }
                
                # Cache the result
                cache_key = self._get_cache_key(window)
                self.attention_cache[cache_key] = window_data.copy()
                
                results.append(window_data)
                logger.info(f"Window {window_idx} processed with QwQ attention")
                
            except Exception as e:
                logger.error(f"Error processing window {window_idx} with QwQ: {e}")
                results.append({
                    'window_idx': window_idx,
                    'text': window,
                    'error': str(e)
                })
        
        return results
    
    def _process_large_window(self, window: str, window_idx: int) -> List[Dict]:
        """Process large windows by chunking them"""
        logger.info(f"Chunking large window {window_idx}")
        
        # Simple chunking by character count
        chunk_size = 25000  # 25K chars per chunk
        chunks = []
        for i in range(0, len(window), chunk_size):
            chunk = window[i:i + chunk_size]
            chunks.append(chunk)
        
        chunk_results = []
        for i, chunk in enumerate(chunks):
            try:
                attention_patterns = self.ollama_extractor.get_attention_patterns(chunk)
                boundary_scores = self.ollama_extractor.analyze_attention_for_boundaries(chunk)
                boundaries = self.ollama_extractor.detect_best_boundaries(chunk, num_segments=3)
                head_specializations = self._detect_qwq_head_specializations(attention_patterns)
                
                chunk_data = {
                    'window_idx': f"{window_idx}_chunk_{i}",
                    'text': chunk,
                    'qwq_attention_patterns': attention_patterns,
                    'boundary_scores': boundary_scores,
                    'optimal_boundaries': boundaries,
                    'head_specializations': head_specializations,
                    'model_info': self.ollama_config,
                    'is_chunk': True,
                    'parent_window': window_idx
                }
                
                chunk_results.append(chunk_data)
                
            except Exception as e:
                logger.error(f"Error processing chunk {i} of window {window_idx}: {e}")
                chunk_results.append({
                    'window_idx': f"{window_idx}_chunk_{i}",
                    'text': chunk,
                    'error': str(e),
                    'is_chunk': True,
                    'parent_window': window_idx
                })
        
        return chunk_results
        
    def analyze_qwq_attention_patterns(self, attention_data: Dict) -> Dict:
        """
        Analyze QwQ attention patterns to identify optimal segmentation points
        
        Args:
            attention_data: Output from extract_attention_for_tape_splitting
            
        Returns:
            Analysis results with segmentation recommendations
        """
        
        analysis = {
            'segmentation_strategy': 'qwq_attention_based',
            'segment_boundaries': [],
            'attention_flow_analysis': {},
            'head_specialization_patterns': {},
            'confidence_scores': []
        }
        
        # Process each window's QwQ attention patterns
        for window_data in attention_data['window_patterns']:
            if 'error' in window_data:
                continue
                
            window_idx = window_data['window_idx']
            
            # Extract segmentation boundaries from QwQ attention
            boundaries = window_data.get('optimal_boundaries', [])
            boundary_scores = window_data.get('boundary_scores', [])
            
            if boundaries:
                analysis['segment_boundaries'].append({
                    'window_idx': window_idx,
                    'boundaries': boundaries,
                    'confidence_scores': boundary_scores,
                    'method': 'qwq_attention_analysis'
                })
                
                # Calculate overall confidence
                if boundary_scores:
                    avg_confidence = sum(boundary_scores) / len(boundary_scores)
                    analysis['confidence_scores'].append(avg_confidence)
            
            # Analyze QwQ head specializations
            head_specs = window_data.get('head_specializations', {})
            if head_specs:
                analysis['head_specialization_patterns'][window_idx] = head_specs
        
        # Calculate overall segmentation confidence
        if analysis['confidence_scores']:
            analysis['overall_confidence'] = sum(analysis['confidence_scores']) / len(analysis['confidence_scores'])
        else:
            analysis['overall_confidence'] = 0.0
            
        return analysis
        
    def _detect_qwq_head_specializations(self, attention_patterns: Dict) -> Dict:
        """
        Detect QwQ-32B attention head specializations for segmentation
        
        Args:
            attention_patterns: Dictionary of layer -> attention tensor from QwQ
            
        Returns:
            Dictionary of QwQ head specializations
        """
        
        specializations = {
            'boundary_detection_heads': [],    # Heads good at finding segment boundaries
            'semantic_relation_heads': [],     # Heads that connect related concepts
            'topic_clustering_heads': [],      # Heads that group similar topics
            'discourse_flow_heads': []         # Heads that track argument flow
        }
        
        if not attention_patterns:
            return specializations
            
        # Analyze each attention head in QwQ's architecture
        for layer_idx, layer_attention in attention_patterns.items():
            if not isinstance(layer_attention, torch.Tensor):
                continue
                
            n_heads = layer_attention.shape[0] if layer_attention.dim() > 2 else 1
            
            for head_idx in range(n_heads):
                head_attention = layer_attention[head_idx] if layer_attention.dim() > 2 else layer_attention
                
                # Calculate specialization scores for this QwQ head
                boundary_score = self._calculate_boundary_detection_score(head_attention)
                relation_score = self._calculate_semantic_relation_score(head_attention)
                clustering_score = self._calculate_topic_clustering_score(head_attention)
                discourse_score = self._calculate_discourse_flow_score(head_attention)
                
                head_id = (int(layer_idx), int(head_idx))
                
                # Assign head to specialization categories (QwQ-specific thresholds)
                if boundary_score > 1.2:  # QwQ threshold for boundary detection
                    specializations['boundary_detection_heads'].append((head_id, boundary_score))
                
                if relation_score > 0.15:  # QwQ threshold for semantic relations
                    specializations['semantic_relation_heads'].append((head_id, relation_score))
                
                if clustering_score > 0.08:  # QwQ threshold for topic clustering
                    specializations['topic_clustering_heads'].append((head_id, clustering_score))
                
                if discourse_score > 0.12:  # QwQ threshold for discourse flow
                    specializations['discourse_flow_heads'].append((head_id, discourse_score))
        
        # Sort and keep top heads for each specialization
        top_k = 8  # Keep top 8 heads per specialization for QwQ
        for spec_type in specializations:
            specializations[spec_type] = sorted(
                specializations[spec_type], 
                key=lambda x: x[1], 
                reverse=True
            )[:top_k]
            
        return specializations
    
    def _calculate_boundary_detection_score(self, attention_matrix: torch.Tensor) -> float:
        """Calculate how well this attention head detects boundaries"""
        if not isinstance(attention_matrix, torch.Tensor):
            return 0.0
        
        # Look for attention patterns that spike at boundaries
        # Simple heuristic: variance in attention weights
        try:
            variance = attention_matrix.var().item()
            return float(variance)
        except:
            return 0.0
    
    def _calculate_semantic_relation_score(self, attention_matrix: torch.Tensor) -> float:
        """Calculate how well this head captures semantic relations"""
        if not isinstance(attention_matrix, torch.Tensor):
            return 0.0
        
        # Look for consistent attention patterns
        try:
            mean_attention = attention_matrix.mean().item()
            return float(mean_attention)
        except:
            return 0.0
    
    def _calculate_topic_clustering_score(self, attention_matrix: torch.Tensor) -> float:
        """Calculate how well this head clusters topics"""
        if not isinstance(attention_matrix, torch.Tensor):
            return 0.0
        
        # Look for block-like attention patterns
        try:
            std_dev = attention_matrix.std().item()
            return float(std_dev)
        except:
            return 0.0
    
    def _calculate_discourse_flow_score(self, attention_matrix: torch.Tensor) -> float:
        """Calculate how well this head tracks discourse flow"""
        if not isinstance(attention_matrix, torch.Tensor):
            return 0.0
        
        # Look for sequential attention patterns
        try:
            # Simple heuristic: check diagonal dominance
            if attention_matrix.dim() >= 2:
                diag_sum = attention_matrix.diagonal().sum().item()
                total_sum = attention_matrix.sum().item()
                return float(diag_sum / total_sum) if total_sum > 0 else 0.0
            return 0.0
        except:
            return 0.0
    
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
def create_attention_extractor(model_name: str = "qwq32b") -> "EnhancedAttentionExtractor":
    """Create an attention extractor for the specified model"""
    return EnhancedAttentionExtractor()