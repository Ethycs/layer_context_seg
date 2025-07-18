#!/usr/bin/env python3
"""
BAAI BGE-EN-ICL Model
=====================
This module provides a unified class for loading the BGE-EN-ICL embedding model
with lazy loading, prompt-based segmentation, and memory-efficient operations.
"""

import gc
import logging
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
from transformers import AutoTokenizer, AutoModel
from .segmenter_protocol import Segmenter

# Import the sharded loader for memory-efficient loading
from utils.safe_tensor_shard_loader import load_sharded_model

logger = logging.getLogger(__name__)

class BAAIModel(Segmenter):
    """
    A class to handle the BGE-EN-ICL model for embeddings and segmentation.
    Uses lazy loading with direct-to-GPU loading for memory efficiency.
    """
    
    def __init__(self, model_path: str, device=None):
        """
        Initialize the model lazily. The model is not loaded until needed.
        
        Args:
            model_path: Path to the BGE-EN-ICL model directory
            device: Device to use (defaults to CUDA if available)
        """
        self.model_path = Path(model_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.max_length = 32768  # BGE-EN-ICL supports up to 32k tokens
        
        # Segmentation parameters
        self.window_size = 512  # Size of sliding window for segmentation
        self.stride = 256  # Stride for sliding window
        
    def _lazy_load(self):
        """
        Loads the model using the memory-efficient sharded loader and the tokenizer.
        """
        if self.model is None:
            logger.info(f"Lazy loading BGE-EN-ICL model from {self.model_path}...")
            
            # Use the sharded loader with the correct AutoModel class
            self.model = load_sharded_model(str(self.model_path), AutoModel)
            
            # Load tokenizer (small, can be loaded normally)
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("BGE-EN-ICL model loaded successfully onto GPU")
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 8, 
               normalize_embeddings: bool = True, show_progress: bool = False, chunk_size: int = 512) -> np.ndarray:
        """
        Encode texts into embeddings using last-token pooling.
        Handles large texts by chunking them.
        """
        self._lazy_load()
        
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=chunk_size,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
                sequence_lengths = inputs.attention_mask.sum(dim=1) - 1
                batch_size_actual = hidden_states.shape[0]
                
                embeddings = hidden_states[range(batch_size_actual), sequence_lengths]
                
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        if not all_embeddings:
            return np.array([])

        embeddings = np.vstack(all_embeddings)
        
        return embeddings
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score
        """
        embeddings = self.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)
    
    def segment_by_prompt(self, text: str, segmentation_prompt: str, 
                         similarity_threshold: float = 0.7) -> List[Dict[str, any]]:
        """
        Segment text based on a guiding prompt using semantic similarity.
        
        Args:
            text: Text to segment
            segmentation_prompt: Prompt describing how to segment
            similarity_threshold: Threshold for determining segment boundaries
            
        Returns:
            List of segments with metadata
        """
        self._lazy_load()
        
        # Tokenize to get total length
        tokens = self.tokenizer.tokenize(text)
        
        if len(tokens) <= self.window_size:
            # Text is short enough to process as single segment
            return [{
                'text': text,
                'start': 0,
                'end': len(text),
                'confidence': 1.0
            }]
        
        # Get prompt embedding to guide segmentation
        prompt_embedding = self.encode(segmentation_prompt)
        
        # Sliding window approach
        windows = []
        embeddings = []
        
        # Create overlapping windows
        for i in range(0, len(tokens) - self.window_size + 1, self.stride):
            window_tokens = tokens[i:i + self.window_size]
            window_text = self.tokenizer.convert_tokens_to_string(window_tokens)
            windows.append({
                'text': window_text,
                'start_token': i,
                'end_token': min(i + self.window_size, len(tokens))
            })
            
            # Get embedding for this window
            window_embedding = self.encode(window_text)
            embeddings.append(window_embedding)
        
        # Stack embeddings for analysis
        embeddings = np.vstack(embeddings)
        
        # Compute similarities between adjacent windows
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        # Also compute similarity to prompt for each window
        prompt_similarities = [np.dot(emb, prompt_embedding) for emb in embeddings]
        
        # Find boundaries where similarity drops
        boundaries = [0]  # Start of text
        
        for i in range(1, len(similarities)):
            # Check if there's a significant drop in similarity
            if similarities[i] < similarity_threshold:
                # Also consider prompt relevance
                if abs(prompt_similarities[i] - prompt_similarities[i-1]) > 0.1:
                    # Convert window index to approximate character position
                    boundary_token = windows[i]['start_token']
                    boundaries.append(boundary_token)
        
        boundaries.append(len(tokens))  # End of text
        
        # Create segments from boundaries
        segments = []
        for i in range(len(boundaries) - 1):
            start_token = boundaries[i]
            end_token = boundaries[i + 1]
            
            segment_tokens = tokens[start_token:end_token]
            segment_text = self.tokenizer.convert_tokens_to_string(segment_tokens)
            
            # Calculate confidence based on internal coherence
            if end_token - start_token > self.window_size:
                # For long segments, sample internal similarity
                mid_point = (start_token + end_token) // 2
                sample1 = tokens[start_token:start_token + 100]
                sample2 = tokens[mid_point:mid_point + 100]
                
                text1 = self.tokenizer.convert_tokens_to_string(sample1)
                text2 = self.tokenizer.convert_tokens_to_string(sample2)
                
                internal_similarity = self.compute_similarity(text1, text2)
                confidence = internal_similarity
            else:
                confidence = 1.0  # Short segments assumed coherent
            
            segments.append({
                'text': segment_text.strip(),
                'start_token': start_token,
                'end_token': end_token,
                'confidence': confidence,
                'prompt_relevance': np.mean([
                    prompt_similarities[j] for j in range(len(windows))
                    if windows[j]['start_token'] >= start_token and windows[j]['end_token'] <= end_token
                ]) if prompt_similarities else 0.5
            })
        
        return segments
    
    def find_semantic_boundaries(self, text: str, min_segment_size: int = 100) -> List[int]:
        """
        Find semantic boundaries in text based on embedding similarity changes.
        
        Args:
            text: Text to analyze
            min_segment_size: Minimum size of segments in characters
            
        Returns:
            List of boundary positions (character indices)
        """
        self._lazy_load()
        
        # Split text into sentences for fine-grained analysis
        sentences = text.split('. ')
        if len(sentences) <= 1:
            return []
        
        # Get embeddings for each sentence
        embeddings = []
        positions = []
        current_pos = 0
        
        for sent in sentences:
            embeddings.append(self.encode(sent))
            positions.append(current_pos)
            current_pos += len(sent) + 2  # +2 for '. '
        
        embeddings = np.vstack(embeddings)
        
        # Compute similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        # Find significant drops in similarity
        boundaries = []
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        threshold = mean_sim - std_sim
        
        last_boundary = 0
        for i, sim in enumerate(similarities):
            if sim < threshold and positions[i+1] - last_boundary >= min_segment_size:
                boundaries.append(positions[i+1])
                last_boundary = positions[i+1]
        
        return boundaries
    
    def unload(self):
        """
        Explicitly unload the model to free GPU memory.
        """
        if self.model is not None:
            logger.info("Unloading BGE-EN-ICL model...")
            self.model = None
            self.tokenizer = None
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Model unloaded and memory cleared")
    
    def reload(self):
        """
        Reload the model after unloading.
        """
        self.unload()
        self._lazy_load()
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model.
        """
        self._lazy_load()
        
        return {
            'model_path': str(self.model_path),
            'device': str(self.device),
            'max_length': self.max_length,
            'embedding_dim': self.model.config.hidden_size,
            'num_layers': self.model.config.num_hidden_layers,
            'num_heads': self.model.config.num_attention_heads,
            'vocab_size': self.model.config.vocab_size,
            'model_type': self.model.config.model_type
        }

    def segment(self, rule: str, text_to_segment: str) -> List[str]:
        """
        Implements the Segmenter protocol by wrapping the find_semantic_boundaries method.
        """
        logger.info(f"BAAIModel segmenting with rule: {rule}")
        boundaries = self.find_semantic_boundaries(text_to_segment)
        
        if not boundaries:
            return [text_to_segment]
            
        segments = []
        start = 0
        for end in boundaries:
            segments.append(text_to_segment[start:end])
            start = end
        segments.append(text_to_segment[start:])
        
        return segments
