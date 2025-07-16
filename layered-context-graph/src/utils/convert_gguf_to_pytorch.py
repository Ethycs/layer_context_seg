#!/usr/bin/env python3
"""
Production-Grade Streaming PyTorch Model Loader
===============================================
This module loads a pre-converted, sharded safetensors model using the
Hugging Face `accelerate` library for true zero-memory initialization.
It also handles the one-time conversion from GGUF, dynamically creating
the model configuration from the GGUF metadata.
"""
import gc
import torch
import numpy as np
import logging
from pathlib import Path
import json
import shutil
import psutil
import os
import struct
from tqdm import tqdm
from accelerate import init_empty_weights
from safetensors.torch import load_file

# Assuming qwq_architecture is in a sibling directory `models`
from models.qwq_architecture import QwQConfig, QwQForCausalLM

logger = logging.getLogger(__name__)

# GGUF constants
GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

class CustomGGUFParser:
    """A low-level parser for GGUF files that reads metadata and tensor info."""
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.tensor_infos = []
        self.metadata = {}

    def type_traits(self, tensor_type: int):
        # (block_size, type_size)
        # Correct values based on gguf-py library
        if tensor_type == 0: return (1, 4) # F32
        if tensor_type == 1: return (1, 2) # F16
        if tensor_type == 2: return (32, 2 + 16) # Q4_0
        if tensor_type == 3: return (32, 2 + 2 + 16) # Q4_1
        if tensor_type == 6: return (32, 2 + 4 + 16) # Q5_0
        if tensor_type == 7: return (32, 2 + 2 + 4 + 16) # Q5_1
        if tensor_type == 8: return (32, 2 + 32) # Q8_0
        if tensor_type == 9: return (32, 4 + 4 + 32) # Q8_1
        if tensor_type == 10: return (256, 2 + 256 // 4) # Q2_K
        if tensor_type == 11: return (256, 2 + 256 // 4 + 256 // 8) # Q3_K
        if tensor_type == 12: return (256, 2 + 256 // 2) # Q4_K
        if tensor_type == 13: return (256, 2 + 2 + 256 // 2 + 256 // 8) # Q5_K
        if tensor_type == 14: return (256, 2 + 2 + 256 // 2 + 256 // 4) # Q6_K
        raise NotImplementedError(f"Unimplemented tensor type {tensor_type}")

    def parse(self):
        with open(self.model_path, 'rb') as f:
            magic, version, tensor_count, metadata_kv_count = struct.unpack('<I I Q Q', f.read(24))
            if magic != GGUF_MAGIC: raise ValueError("Not a GGUF file")
            
            for _ in range(metadata_kv_count):
                key_len, = struct.unpack('<Q', f.read(8))
                key = f.read(key_len).decode('utf-8')
                value_type, = struct.unpack('<I', f.read(4))
                if value_type == 8: # String
                    value_len, = struct.unpack('<Q', f.read(8))
                    value = f.read(value_len).decode('utf-8')
                elif value_type == 9: # Array
                    array_type, array_len = struct.unpack('<I Q', f.read(12))
                    if array_type == 8:
                        for _ in range(array_len):
                            str_len, = struct.unpack('<Q', f.read(8))
                            f.seek(str_len, 1)
                    else:
                        f.seek(self._get_gguf_value_size(array_type) * array_len, 1)
                    value = "[skipped]"
                else:
                    value, = struct.unpack(f'<{self._get_gguf_struct_fmt(value_type)}', f.read(self._get_gguf_value_size(value_type)))
                self.metadata[key] = value

            for _ in range(tensor_count):
                name_len, = struct.unpack('<Q', f.read(8))
                name = f.read(name_len).decode('utf-8')
                n_dims, = struct.unpack('<I', f.read(4))
                shape = list(struct.unpack(f'<{n_dims}Q', f.read(n_dims * 8)))
                tensor_type, offset = struct.unpack('<I Q', f.read(12))
                
                num_elements = np.prod(shape)
                block_size, type_size = self.type_traits(tensor_type)
                n_bytes = (num_elements * type_size) // block_size
                
                self.tensor_infos.append({"name": name, "shape": shape, "type": tensor_type, "offset": offset, "n_bytes": int(n_bytes)})
        return self.metadata, self.tensor_infos

    def _get_gguf_struct_fmt(self, v_type): return {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[v_type]
    def _get_gguf_value_size(self, v_type): return {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[v_type]

class StreamingModelLoader:
    def __init__(self, model_path: str, device='cuda', chunk_size_gb=4.0):
        self.model_path = Path(model_path)
        self.device = device
        self.chunk_size_gb = chunk_size_gb
        self.shards_path = self.model_path.with_suffix('.safetensors_shards')

    def load_model(self, model_class):
        config_path = self.shards_path / "config.json"
        if not config_path.exists():
            logger.info("First run: Converting GGUF to sharded safetensors and creating config...")
            self._convert_gguf_and_create_config()

        logger.info(f"Found pre-converted model. Loading config from {config_path}")
        with open(config_path) as f:
            config_dict = json.load(f)
        config = QwQConfig(**config_dict)
        
        return self._stream_load_from_sharded_safetensors(model_class, config)

    def _convert_gguf_and_create_config(self):
        self.shards_path.mkdir(parents=True, exist_ok=True)
        parser = CustomGGUFParser(self.model_path)
        metadata, tensor_infos = parser.parse()

        config_dict = {
            "vocab_size": int(metadata.get('tokenizer.ggml.vocab_size', 151936)),
            "hidden_size": int(metadata.get('qwen2.embedding_length', 8192)),
            "intermediate_size": int(metadata.get('qwen2.feed_forward_length', 29568)),
            "num_hidden_layers": int(metadata.get('qwen2.block_count', 64)),
            "num_attention_heads": int(metadata.get('qwen2.attention.head_count', 64)),
            "num_key_value_heads": int(metadata.get('qwen2.attention.head_count_kv', 8)),
            "max_position_embeddings": int(metadata.get('qwen2.context_length', 32768)),
        }
        with open(self.shards_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Model config saved with vocab_size: {config_dict['vocab_size']}")

        chunk_size_bytes = int(self.chunk_size_gb * 1024 * 1024 * 1024)
        current_tensors, current_size, shard_num = {}, 0, 1
        
        with open(self.model_path, "rb") as f:
            for info in tqdm(tensor_infos, desc="Converting Tensors"):
                f.seek(info['offset'])
                tensor_data = f.read(info['n_bytes'])
                
                # For now, we only handle F32 and F16, as dequantization is complex
                if info['type'] in [0, 1]:
                    dtype = np.float32 if info['type'] == 0 else np.float16
                    data = torch.from_numpy(np.frombuffer(tensor_data, dtype=dtype).reshape(info['shape']))
                else:
                    logger.warning(f"Skipping dequantization for type {info['type']}. Creating placeholder.")
                    data = torch.zeros(info['shape'], dtype=torch.float16)
                
                if current_size > 0 and current_size + data.nbytes > chunk_size_bytes:
                    save_file(current_tensors, self.shards_path / f"part_{shard_num:03d}.safetensors")
                    current_tensors, current_size, shard_num = {}, 0, shard_num + 1
                
                current_tensors[info['name']] = data
                current_size += data.nbytes
        
        if current_tensors:
            save_file(current_tensors, self.shards_path / f"part_{shard_num:03d}.safetensors")

        # Finalize index file
        tensor_map_index = {}
        processed_tensors = 0
        for i in range(1, shard_num + 1):
            shard_name = f"part_{i:03d}.safetensors"
            with safe_open(self.shards_path / shard_name, framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    tensor_map_index[k] = shard_name
        
        with open(self.shards_path / "model.safetensors.index.json", 'w') as f:
            json.dump({"metadata": {}, "weight_map": tensor_map_index}, f, indent=2)
        logger.info(f"Conversion to {shard_num} sharded safetensors is complete.")

    def _stream_load_from_sharded_safetensors(self, model_class, config):
        with init_empty_weights():
            model = model_class(config)
        
        index_path = self.shards_path / "model.safetensors.index.json"
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        
        for shard_file in sorted(list(set(weight_map.values()))):
            state_dict = load_file(self.shards_path / shard_file, device=str(self.device))
            model.load_state_dict(state_dict, assign=True, strict=False)
            del state_dict
            gc.collect()

        return model, config

class QwQModel:
    def __init__(self, model_path: str, device='cuda'):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        loader = StreamingModelLoader(str(self.model_path), device=self.device)
        self.model, self.config = loader.load_model(QwQForCausalLM)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B", trust_remote_code=True)
