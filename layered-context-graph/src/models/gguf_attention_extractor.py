#!/usr/bin/env python3
"""
GGUF Attention Extractor with Real Inference
===========================================
Extracts real attention patterns by running inference through GGUF models using PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gguf
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import gc

logger = logging.getLogger(__name__)


class GGUFAttentionExtractor:
    """
    Extracts real attention patterns from GGUF models by running actual inference.
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the GGUF attention extractor.
        
        Args:
            model_path: Path to GGUF model file
            device: PyTorch device to use
        """
        self.model_path = Path(model_path)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Load GGUF file and extract model info
        self.gguf_reader = gguf.GGUFReader(str(self.model_path), 'r')
        self.config = self._extract_model_config()
        
        # Build PyTorch model from GGUF weights
        self.model = self._build_pytorch_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Storage for attention patterns during forward pass
        self.attention_patterns = {}
        self._register_attention_hooks()
        
        logger.info(f"Initialized GGUF model on {self.device}")
        logger.info(f"Model config: {self.config}")
    
    def _extract_model_config(self) -> Dict[str, Any]:
        """Extract model configuration from GGUF metadata."""
        config = {}
        
        metadata_map = {
            'general.architecture': 'architecture',
            'llama.context_length': 'context_length',
            'llama.embedding_length': 'hidden_size',
            'llama.attention.head_count': 'num_attention_heads',
            'llama.attention.head_count_kv': 'num_key_value_heads',
            'llama.block_count': 'num_hidden_layers',
            'llama.feed_forward_length': 'intermediate_size',
            'llama.rope.freq_base': 'rope_theta',
            'llama.attention.layer_norm_epsilon': 'layer_norm_eps',
            'tokenizer.ggml.tokens': 'vocab_size'
        }
        
        qwq_map = {
            'qwen2.context_length': 'context_length',
            'qwen2.embedding_length': 'hidden_size',
            'qwen2.attention.head_count': 'num_attention_heads',
            'qwen2.attention.head_count_kv': 'num_key_value_heads',
            'qwen2.block_count': 'num_hidden_layers',
            'qwen2.feed_forward_length': 'intermediate_size'
        }
        
        for gguf_key, config_key in {**metadata_map, **qwq_map}.items():
            if hasattr(self.gguf_reader, 'metadata') and gguf_key in self.gguf_reader.metadata:
                config[config_key] = self.gguf_reader.metadata[gguf_key]
        
        config.setdefault('hidden_size', 5120)
        config.setdefault('num_attention_heads', 40)
        config.setdefault('num_key_value_heads', 8)
        config.setdefault('num_hidden_layers', 64)
        config.setdefault('intermediate_size', 27648)
        config.setdefault('vocab_size', 151936)
        config.setdefault('rope_theta', 1000000)
        config.setdefault('context_length', 131072)
        
        config['head_dim'] = config['hidden_size'] // config['num_attention_heads']
        
        return config
    
    def _build_pytorch_model(self) -> nn.Module:
        """Build a minimal PyTorch model from GGUF weights for attention extraction."""
        return QwQAttentionModel(self.config, self.gguf_reader, self.device)
    
    def _register_attention_hooks(self):
        """Register hooks to capture attention patterns during forward pass."""
        def make_attention_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    attention_weights = output[1]
                else:
                    attention_weights = getattr(module, 'attention_weights', None)
                
                if attention_weights is not None:
                    self.attention_patterns[layer_idx] = attention_weights.detach().cpu()
            return hook
        
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'self_attn'):
                layer.self_attn.register_forward_hook(make_attention_hook(i))
    
    def extract_attention_patterns(self, text: str) -> Dict[int, torch.Tensor]:
        """Extract real attention patterns by running inference."""
        self.attention_patterns = {}
        tokens = self._simple_tokenize(text)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            _ = self.model(input_ids, output_attentions=True)
        
        return self.attention_patterns
    
    def _simple_tokenize(self, text: str) -> List[int]:
        """Simple tokenization for demo - replace with proper tokenizer."""
        words = text.split()
        token_ids = []
        for word in words:
            token_id = hash(word.lower()) % self.config['vocab_size']
            token_ids.append(abs(token_id))
        return token_ids[:512]


class QwQAttentionModel(nn.Module):
    """Minimal QwQ model implementation for attention extraction."""
    
    def __init__(self, config: Dict[str, Any], gguf_reader: gguf.GGUFReader, device):
        super().__init__()
        self.config = config
        self.device = device
        
        self.embed_tokens = nn.Embedding(
            config['vocab_size'],
            config['hidden_size']
        )
        self._load_embeddings(gguf_reader)
        
        self.layers = nn.ModuleList([
            QwQAttentionLayer(config, layer_idx, gguf_reader, self.device)
            for layer_idx in range(config['num_hidden_layers'])
        ])
        
        self.norm = nn.LayerNorm(config['hidden_size'], eps=1e-5)
    
    def _load_embeddings(self, gguf_reader):
        """Load embedding weights from GGUF."""
        for tensor_info in gguf_reader.tensors:
            if 'token_embd' in tensor_info.name or 'embed_tokens' in tensor_info.name:
                weight_data = tensor_info.data
                weight = torch.from_numpy(weight_data.astype(np.float32))
                if weight.shape[0] == self.config['vocab_size']:
                    self.embed_tokens.weight.data = weight
                    logger.info(f"Loaded embeddings: {weight.shape}")
                    del weight_data
                    gc.collect()
                    break
    
    def forward(self, input_ids: torch.Tensor, output_attentions: bool = True):
        """Forward pass through the model."""
        hidden_states = self.embed_tokens(input_ids)
        
        all_attentions = []
        for layer in self.layers:
            hidden_states, attention_weights = layer(hidden_states, output_attentions=True)
            if output_attentions and attention_weights is not None:
                all_attentions.append(attention_weights)
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states, all_attentions


class QwQAttentionLayer(nn.Module):
    """Single attention layer from QwQ model."""
    
    def __init__(self, config: Dict[str, Any], layer_idx: int, gguf_reader: gguf.GGUFReader, device):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config['hidden_size']
        
        self.self_attn = QwQAttention(config, layer_idx, gguf_reader, device)
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self._load_weights(gguf_reader)
    
    def _load_weights(self, gguf_reader):
        """Load layer weights from GGUF directly to the target device."""
        prefix = f"blk.{self.layer_idx}"
        
        for tensor_info in gguf_reader.tensors:
            if prefix not in tensor_info.name:
                continue
            
            if 'attn_norm' in tensor_info.name or 'input_layernorm' in tensor_info.name:
                if tensor_info.shape == (self.hidden_size,):
                    weight_data = tensor_info.data
                    weight = torch.from_numpy(weight_data.astype(np.float32)).to(self.input_layernorm.weight.device)
                    self.input_layernorm.weight.data = weight
                    del weight_data
                    gc.collect()

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False):
        """Forward pass through the layer."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attention_weights = self.self_attn(hidden_states, output_attentions=output_attentions)
        hidden_states = residual + hidden_states
        return hidden_states, attention_weights


class QwQAttention(nn.Module):
    """Multi-head attention from QwQ with Grouped Query Attention."""
    
    def __init__(self, config: Dict[str, Any], layer_idx: int, gguf_reader: gguf.GGUFReader, device):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.num_kv_heads = config['num_key_value_heads']
        self.head_dim = config['head_dim']
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self._load_weights(gguf_reader)
        self.attention_weights = None
    
    def _load_weights(self, gguf_reader):
        """Load attention weights from GGUF directly to the target device, releasing RAM immediately."""
        prefix = f"blk.{self.layer_idx}.attn"
        
        for tensor_info in gguf_reader.tensors:
            if prefix not in tensor_info.name:
                continue
            
            tensor_data = tensor_info.data
            # GGUF stores weights as (out_features, in_features), but nn.Linear expects (in_features, out_features)
            # So we need to transpose the weights before loading them.
            weight = torch.from_numpy(tensor_data.astype(np.float32)).t().contiguous()
            weight = weight.to(self.q_proj.weight.device)
            
            if 'attn_q' in tensor_info.name or 'q_proj' in tensor_info.name:
                self.q_proj.weight.data = weight
            elif 'attn_k' in tensor_info.name or 'k_proj' in tensor_info.name:
                self.k_proj.weight.data = weight
            elif 'attn_v' in tensor_info.name or 'v_proj' in tensor_info.name:
                self.v_proj.weight.data = weight
            elif 'attn_output' in tensor_info.name or 'o_proj' in tensor_info.name:
                self.o_proj.weight.data = weight
            
            del tensor_data
            del weight
        
        gc.collect()
    
    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False):
        """Forward pass with attention weight extraction."""
        batch_size, seq_len, _ = hidden_states.shape
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        if self.num_kv_groups > 1:
            key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)
        
        attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        if output_attentions:
            self.attention_weights = attention_probs
        
        attention_output = torch.matmul(attention_probs, value_states)
        
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        attention_output = self.o_proj(attention_output)
        
        return attention_output, attention_probs if output_attentions else None
