"""
QwQ-32B Attention Extractor - Primary model for layered context segmentation
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging

from models.qwq_model import QwQModel

logger = logging.getLogger(__name__)


class EnhancedAttentionExtractor:
    """
    QwQ-32B attention extractor for layered context segmentation.
    This is the primary and only model used for tape splitting.
    """
    
    def __init__(self, qwq_model: QwQModel):
        """
        Initialize the QwQ attention extractor
        
        Args:
            qwq_model: An existing QwQModel instance to reuse
        """
        if not qwq_model:
            raise ValueError("A pre-loaded QwQModel instance is required.")
            
        self.qwq_model = qwq_model
        self.attention_cache = {}
        
        # For compatibility, some methods might expect an 'ollama_extractor' attribute
        self.ollama_extractor = self.qwq_model
        self.ollama_config = self.qwq_model.get_model_config()
        
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
        
    def extract_attention_for_tape_splitting(self, text_windows: List[str]) -> Dict:
        """
        Extract attention patterns from text windows using the unified QwQModel.
        """
        logger.info(f"Extracting attention patterns for {len(text_windows)} windows.")
        
        all_patterns = []
        for i, window in enumerate(text_windows):
            attention_patterns = self.qwq_model.extract_attention(window)
            window_data = {
                'window_idx': i,
                'text': window,
                'qwq_attention_patterns': attention_patterns,
                'model_info': self.ollama_config
            }
            all_patterns.append(window_data)

        return {
            'model': 'qwq-32b',
            'segmentation_method': 'qwq_attention_heads',
            'window_patterns': all_patterns,
            'model_config': self.ollama_config
        }
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for text content"""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
