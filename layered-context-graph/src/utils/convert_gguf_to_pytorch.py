#!/usr/bin/env python3
"""
GGUF to PyTorch Model Converter
===============================
This script converts a GGUF model file into a PyTorch state_dict (.bin) file,
allowing it to be loaded with memory-mapping for reduced RAM usage.
"""

import logging
import torch
import gguf
from pathlib import Path
import numpy as np

# Add src directory to path to import architecture
import sys
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from models.qwq_architecture import QwQConfig, QwQForCausalLM

logger = logging.getLogger(__name__)

def get_tensor_name_map():
    """
    Returns the mapping from GGUF tensor names to PyTorch state_dict keys.
    """
    mapping = {
        'token_embd.weight': 'model.embed_tokens.weight',
        'output_norm.weight': 'model.norm.weight',
        'output.weight': 'lm_head.weight',
    }
    
    # Programmatically generate mappings for all decoder layers
    for i in range(64): # Assuming 64 layers for QwQ-32B
        mapping[f'blk.{i}.attn_norm.weight'] = f'model.layers.{i}.input_layernorm.weight'
        mapping[f'blk.{i}.ffn_norm.weight'] = f'model.layers.{i}.post_attention_layernorm.weight'
        mapping[f'blk.{i}.attn_q.weight'] = f'model.layers.{i}.self_attn.q_proj.weight'
        mapping[f'blk.{i}.attn_k.weight'] = f'model.layers.{i}.self_attn.k_proj.weight'
        mapping[f'blk.{i}.attn_v.weight'] = f'model.layers.{i}.self_attn.v_proj.weight'
        mapping[f'blk.{i}.attn_output.weight'] = f'model.layers.{i}.self_attn.o_proj.weight'
        mapping[f'blk.{i}.ffn_gate.weight'] = f'model.layers.{i}.mlp.gate_proj.weight'
        mapping[f'blk.{i}.ffn_up.weight'] = f'model.layers.{i}.mlp.up_proj.weight'
        mapping[f'blk.{i}.ffn_down.weight'] = f'model.layers.{i}.mlp.down_proj.weight'
        
    return mapping

def convert_gguf_to_pytorch(gguf_path: str, output_path: str):
    """
    Converts a GGUF file to a PyTorch state_dict file.
    """
    logger.info(f"Starting conversion of {gguf_path} to {output_path}")
    
    try:
        reader = gguf.GGUFReader(gguf_path, 'r')
        tensor_map = get_tensor_name_map()
        
        state_dict = {}
        unmapped_tensors = []

        for tensor in reader.tensors:
            if tensor.name in tensor_map:
                pytorch_key = tensor_map[tensor.name]
                # Ensure data is in a writable numpy array before converting to tensor
                state_dict[pytorch_key] = torch.from_numpy(np.array(tensor.data))
            else:
                unmapped_tensors.append(tensor.name)
        
        if unmapped_tensors:
            logger.warning(f"Found {len(unmapped_tensors)} unmapped tensors. This may be expected.")
            logger.debug(f"Unmapped tensors: {unmapped_tensors[:10]}...")

        logger.info(f"Saving converted state_dict with {len(state_dict)} tensors to {output_path}")
        torch.save(state_dict, output_path)
        logger.info("Conversion successful.")

    except Exception as e:
        logger.error(f"Failed to convert GGUF model: {e}")
        raise

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).parent.parent.parent
    gguf_file = project_root / "qwq.gguf"
    pytorch_file = project_root / "models" / "qwq_pytorch.bin"
    
    if not gguf_file.exists():
        logger.error(f"GGUF file not found at {gguf_file}. Please place it in the project root.")
    else:
        convert_gguf_to_pytorch(str(gguf_file), str(pytorch_file))
