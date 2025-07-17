# save as convert_partial.py
# Ref https://github.com/huggingface/transformers/blob/b85ed49e0a5f1bd9fd887f497d055b22b9319a12/src/transformers/modeling_gguf_pytorch_utils.py#L55
import os
import re
from typing import NamedTuple, Optional
import numpy as np
from tqdm.auto import tqdm

# --- User Configuration ---
# 1. Set the path to your large GGUF file
GGUF_INPUT_PATH = "qwq.gguf" 

# 2. Set the desired path for the output safetensors file
SAFETENSORS_OUTPUT_PATH = "qwq-shard-0.safetensors"

# 3. Set the maximum memory (in Gigabytes) to use for the dequantized tensors
MEMORY_LIMIT_GB = 4.0
# --------------------------


# --- Dependencies Check ---
try:
    import torch
    from gguf import GGUFReader, dequantize
    from safetensors.torch import save_file
except ImportError:
    print("Error: This script requires torch, gguf, and safetensors.")
    print("Please install them with: pip install torch gguf safetensors")
    exit()

# We will use a simplified version of the loader focused only on tensor conversion
def convert_gguf_to_partial_safetensors(gguf_path, output_path, max_gb):
    """
    Reads tensors from a GGUF file, dequantizes them until a memory limit is reached,
    and saves the result to a safetensors file.
    """
    if not os.path.exists(gguf_path):
        print(f"Error: Input GGUF file not found at: {gguf_path}")
        return

    print(f"Reading GGUF file: {gguf_path}")
    reader = GGUFReader(gguf_path)
    
    tensors_to_save = {}
    current_size_bytes = 0
    limit_bytes = max_gb * (1024**3)

    print(f"Starting conversion with a memory limit of {max_gb} GB...")
    
    # This is the core loop from the file you provided, now with a memory limit
    for tensor_info in tqdm(reader.tensors, desc="Converting and de-quantizing GGUF tensors..."):
        # Dequantize the tensor from the file
        weights = dequantize(tensor_info.data, tensor_info.tensor_type)
        
        # Convert to a PyTorch tensor
        # NOTE: We use the raw tensor name from GGUF. A full conversion would map this
        # to the Hugging Face name, but for sharding, the raw name is sufficient.
        tensor_name = tensor_info.name
        torch_tensor = torch.from_numpy(np.copy(weights))
        
        # Check if adding this tensor would exceed the memory limit
        if current_size_bytes + torch_tensor.nbytes > limit_bytes:
            print(f"\nMemory limit reached. Stopping before adding tensor: {tensor_name}")
            print(f"Total size of tensors to be saved: {current_size_bytes / (1024**3):.2f} GB")
            break
            
        # Add the tensor to our dictionary and update the current size
        tensors_to_save[tensor_name] = torch_tensor
        current_size_bytes += torch_tensor.nbytes

    if not tensors_to_save:
        print("No tensors were converted. The memory limit might be too low or the file is empty.")
        return

    # Save the collected tensors to a safetensors file
    print(f"\nSaving {len(tensors_to_save)} tensors to {output_path}...")
    try:
        save_file(tensors_to_save, output_path)
        print("Successfully saved partial model shard.")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


if __name__ == "__main__":
    convert_gguf_to_partial_safetensors(
        gguf_path=GGUF_INPUT_PATH,
        output_path=SAFETENSORS_OUTPUT_PATH,
        max_gb=MEMORY_LIMIT_GB
    )

