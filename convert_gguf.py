import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
from safetensors.torch import save_file
from tqdm import tqdm
import gguf
import os
import mmap
from dataclasses import dataclass
import struct
import threading

class IncrementalFileBuilder:
    """
    Build a large file incrementally from smaller buffers.
    Flushes data as we go to avoid memory buildup.
    """
    
    def __init__(self, output_path: str, size: int):
        self.output_path = output_path
        self.size = size
        
        # Create file
        with open(output_path, 'wb') as f:
            f.truncate(size)
        
        # Open file and mmap
        self.file = open(output_path, 'r+b')
        self.mmap = mmap.mmap(self.file.fileno(), size, access=mmap.ACCESS_WRITE)
        
    def write(self, offset, data):
        """Write data to a specific region"""
        self.mmap.seek(offset)
        self.mmap.write(data)
        
    def close(self):
        self.mmap.flush()
        self.mmap.close()
        self.file.close()

def convert_gguf_to_safetensors(gguf_path: str, output_dir: str):
    """Convert GGUF to PyTorch format"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    reader = gguf.GGUFReader(gguf_path, 'r')
    
    config_dict = {
        "vocab_size": 151936,
        "hidden_size": int(reader.fields['qwen2.embedding_length'].parts[0].item()),
        "intermediate_size": int(reader.fields['qwen2.feed_forward_length'].parts[0].item()),
        "num_hidden_layers": int(reader.fields['qwen2.block_count'].parts[0].item()),
        "num_attention_heads": int(reader.fields['qwen2.attention.head_count'].parts[0].item()),
        "num_key_value_heads": int(reader.fields['qwen2.attention.head_count_kv'].parts[0].item()),
        "max_position_embeddings": int(reader.fields['qwen2.context_length'].parts[0].item()),
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
        
    output_path = output_dir / "model.safetensors"
    
    header = {"__metadata__": {}}
    offset = 0
    
    # First pass to calculate total size and build header
    for tensor in reader.tensors:
        tensor_array = tensor.data
        pytorch_tensor = torch.from_numpy(tensor_array).to(torch.float16)
        header[tensor.name] = {
            "dtype": "F16",
            "shape": [int(x) for x in tensor.shape],
            "data_offsets": [offset, offset + pytorch_tensor.nbytes],
        }
        offset += pytorch_tensor.nbytes
        
    header_bytes = json.dumps(header).encode('utf-8')
    header_size = len(header_bytes)
    total_size = header_size + 8 + offset
    
    buffer = IncrementalFileBuilder(output_path, total_size)
    
    # Write header
    buffer.write(0, struct.pack('<Q', header_size))
    buffer.write(8, header_bytes)
    
    # Write tensors
    offset = header_size + 8
    for tensor in tqdm(reader.tensors, desc="Converting Tensors"):
        tensor_array = tensor.data
        pytorch_tensor = torch.from_numpy(tensor_array).to(torch.float16)
        tensor_bytes = pytorch_tensor.numpy().tobytes()
        buffer.write(offset, tensor_bytes)
        offset += len(tensor_bytes)
        
    buffer.close()
    
if __name__ == "__main__":
    convert_gguf_to_safetensors("qwq.gguf", "./qwq_safetensors")
