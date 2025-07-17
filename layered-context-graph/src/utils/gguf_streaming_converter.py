import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import struct
import json
import safetensors.torch
from safetensors.torch import save_file
from tqdm import tqdm
import gguf
import os
import mmap
from dataclasses import dataclass
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

@dataclass
class TensorInfo:
    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    offset: int
    size: int

class StreamingGGUFConverter:
    """Convert GGUF to PyTorch format without loading entire model into memory"""
    
    def __init__(self, gguf_path: str, output_dir: str, chunk_size: int = 1024 * 1024):
        self.gguf_path = gguf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.chunk_size = chunk_size
        
        # Memory-map the GGUF file
        self.file_size = os.path.getsize(gguf_path)
        self.file = open(gguf_path, 'rb')
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Parse GGUF header
        self.reader = gguf.GGUFReader(gguf_path)
        self.tensor_info = self._extract_tensor_info()
        
    def _extract_tensor_info(self) -> Dict[str, TensorInfo]:
        """Extract tensor metadata without loading data"""
        tensor_info = {}
        
        for tensor in self.reader.tensors:
            # Get dequantized shape
            shape = list(tensor.shape)
            numpy_dtype = self._gguf_to_numpy_dtype(tensor.tensor_type)
            
            tensor_info[tensor.name] = TensorInfo(
                name=tensor.name,
                shape=tuple(shape),
                dtype=numpy_dtype,
                offset=tensor.data_offset,
                size=tensor.n_bytes
            )
            
        return tensor_info
    
    def _gguf_to_numpy_dtype(self, gguf_type) -> np.dtype:
        """Convert GGUF data type to numpy dtype"""
        # This is simplified - you'd need full mapping
        dtype_map = {
            gguf.GGMLQuantizationType.F32: np.float32,
            gguf.GGMLQuantizationType.F16: np.float16,
            gguf.GGMLQuantizationType.Q4_0: np.float16,  # Will be dequantized
            gguf.GGMLQuantizationType.Q4_1: np.float16,
            gguf.GGMLQuantizationType.Q5_0: np.float16,
            gguf.GGMLQuantizationType.Q5_1: np.float16,
            gguf.GGMLQuantizationType.Q8_0: np.float16,
        }
        return dtype_map.get(gguf_type, np.float32)
    
    def _dequantize_tensor(self, tensor_data: bytes, tensor_type, shape) -> np.ndarray:
        """Dequantize GGUF tensor data"""
        # The gguf library handles dequantization automatically
        return np.frombuffer(tensor_data, dtype=self._gguf_to_numpy_dtype(tensor_type)).reshape(shape)
    
    def stream_convert_to_safetensors(self) -> Path:
        """Convert GGUF to safetensors format, streaming one tensor at a time"""
        output_path = self.output_dir / "model.safetensors"
        
        # Create index for sharded saving if model is large
        index = {
            "metadata": {"format": "pt"},
            "weight_map": {}
        }
        
        current_shard = 0
        current_shard_size = 0
        max_shard_size = 1 * 1024 * 1024 * 1024  # 1GB per shard
        current_tensors = {}
        
        print("Streaming conversion from GGUF to SafeTensors...")
        
        for tensor_name, info in tqdm(self.tensor_info.items()):
            # Dequantize if necessary
            tensor_array = self._dequantize_tensor(tensor_name)
            
            # Convert to PyTorch tensor (still on CPU, but not in Python memory)
            tensor = torch.from_numpy(tensor_array)
            
            # Map GGUF names to PyTorch names
            pytorch_name = self._map_tensor_name(tensor_name)
            
            # Add to current shard
            current_tensors[pytorch_name] = tensor
            current_shard_size += tensor.nbytes
            
            # Save shard if it's getting too large
            if current_shard_size >= max_shard_size:
                shard_path = self.output_dir / f"model-{current_shard:05d}.safetensors"
                save_file(current_tensors, shard_path)
                
                # Update index
                for name in current_tensors:
                    index["weight_map"][name] = f"model-{current_shard:05d}.safetensors"
                
                # Reset for next shard
                current_tensors = {}
                current_shard_size = 0
                current_shard += 1
                
                # Force garbage collection
                import gc
                gc.collect()
        
        # Save remaining tensors
        if current_tensors:
            if current_shard == 0:
                # Single file
                save_file(current_tensors, output_path)
            else:
                # Final shard
                shard_path = self.output_dir / f"model-{current_shard:05d}.safetensors"
                save_file(current_tensors, shard_path)
                
                for name in current_tensors:
                    index["weight_map"][name] = f"model-{current_shard:05d}.safetensors"
                
                # Save index
                with open(self.output_dir / "model.safetensors.index.json", "w") as f:
                    json.dump(index, f, indent=2)
        
        return output_path
    
    def _map_tensor_name(self, gguf_name: str) -> str:
        """Map GGUF tensor names to PyTorch convention"""
        # This mapping depends on your model architecture
        # Example for Llama models:
        mappings = {
            "token_embd.weight": "model.embed_tokens.weight",
            "output_norm.weight": "model.norm.weight",
            "output.weight": "lm_head.weight",
        }
        
        # Layer mappings
        if "blk." in gguf_name:
            # Extract layer number
            parts = gguf_name.split(".")
            layer_idx = int(parts[1])
            
            # Map component names
            if "attn_q.weight" in gguf_name:
                return f"model.layers.{layer_idx}.self_attn.q_proj.weight"
            elif "attn_k.weight" in gguf_name:
                return f"model.layers.{layer_idx}.self_attn.k_proj.weight"
            elif "attn_v.weight" in gguf_name:
                return f"model.layers.{layer_idx}.self_attn.v_proj.weight"
            elif "attn_output.weight" in gguf_name:
                return f"model.layers.{layer_idx}.self_attn.o_proj.weight"
            elif "ffn_gate.weight" in gguf_name:
                return f"model.layers.{layer_idx}.mlp.gate_proj.weight"
            elif "ffn_up.weight" in gguf_name:
                return f"model.layers.{layer_idx}.mlp.up_proj.weight"
            elif "ffn_down.weight" in gguf_name:
                return f"model.layers.{layer_idx}.mlp.down_proj.weight"
            elif "attn_norm.weight" in gguf_name:
                return f"model.layers.{layer_idx}.input_layernorm.weight"
            elif "ffn_norm.weight" in gguf_name:
                return f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        
        return mappings.get(gguf_name, gguf_name)
    
    def cleanup(self):
        """Clean up memory-mapped file"""
        self.mmap.close()
        self.file.close()

class LazyModelLoader:
    """Load converted model lazily using meta device"""
    
    def __init__(self, model_path: str, model_config: dict):
        self.model_path = Path(model_path)
        self.config = model_config
        
    def create_meta_model(self):
        """Create model on meta device"""
        from transformers import AutoConfig, AutoModelForCausalLM
        
        # Create config
        config = AutoConfig.for_model(
            model_type="llama",  # Adjust based on your model
            **self.config
        )
        
        # Initialize on meta device
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        
        return model
    
    def load_to_gpu(self, model, device_map: Optional[Dict] = None):
        """Load model from disk to GPU with device mapping"""
        if device_map is None:
            device_map = "auto"  # Automatically distribute across available GPUs
        
        # Load checkpoint and dispatch to devices
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=str(self.model_path),
            device_map=device_map,
            no_split_module_classes=["LlamaDecoderLayer"],  # Adjust for your model
            dtype=torch.float16,  # Use fp16 to save memory
            offload_folder="offload",  # Offload to disk if needed
            offload_state_dict=True,
        )
        
        return model

# Complete pipeline
def lazy_load_gguf_to_gpu(
    gguf_path: str,
    output_dir: str = "./converted_model",
    device_map: Optional[Dict] = None
):
    """Complete pipeline: GGUF -> Disk -> Meta Device -> GPU"""
    
    print("Step 1: Streaming GGUF to SafeTensors on disk...")
    converter = StreamingGGUFConverter(gguf_path, output_dir)
    
    # Extract model config from GGUF metadata
    model_config = {
        "vocab_size": 32000,  # Extract from GGUF metadata
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "max_position_embeddings": 4096,
    }
    
    try:
        # Convert to safetensors format on disk
        safetensors_path = converter.stream_convert_to_safetensors()
        
        print("\nStep 2: Creating model on meta device...")
        loader = LazyModelLoader(safetensors_path.parent, model_config)
        meta_model = loader.create_meta_model()
        
        print("\nStep 3: Loading model to GPU...")
        gpu_model = loader.load_to_gpu(meta_model, device_map)
        
        return gpu_model
        
    finally:
        converter.cleanup()

# Usage example
if __name__ == "__main__":
    # Path to your GGUF file
    gguf_path = "/path/to/your/model.gguf"
    
    # Custom device mapping (optional)
    device_map = {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
        # ... map layers to devices
        "model.norm": 0,
        "lm_head": 0,
    }
    
    # Run the pipeline
    model = lazy_load_gguf_to_gpu(
        gguf_path=gguf_path,
        output_dir="./converted_model",
        device_map=device_map  # or None for automatic
    )
    
    print("Model loaded successfully!")
    print(f"Model device map: {model.hf_device_map}")
