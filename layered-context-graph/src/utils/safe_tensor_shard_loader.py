from accelerate import init_empty_weights
import torch
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from safetensors import safe_open
import json

class ShardedSafeTensorsLoader:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        with open(self.model_dir / "model.safetensors.index.json") as f:
            self.index = json.load(f)
    
    def load_to_gpu(self, model):
        """Load sharded SafeTensors directly to GPU"""
        weight_map = self.index["weight_map"]
        
        # Group by shard
        shards = {}
        for param_name, shard_file in weight_map.items():
            if shard_file not in shards:
                shards[shard_file] = []
            shards[shard_file].append(param_name)
        
        # Load each shard
        for shard_file, param_names in shards.items():
            shard_path = self.model_dir / shard_file
            
            # Open SafeTensors file with direct GPU loading
            with safe_open(shard_path, framework="pt", device="cuda:0") as f:
                for param_name in param_names:
                    if param_name in f.keys():
                        # Direct GPU load - no RAM used!
                        tensor = f.get_tensor(param_name)
                        self._assign_param(model, param_name, tensor)
            # Clean up CUDA cache between shards
            torch.cuda.empty_cache()
        return model
            

    
    def _assign_param(self, model, param_name, tensor):
        """Assign parameter to model"""
        keys = param_name.split('.')
        ptr = model
        for key in keys[:-1]:
            ptr = getattr(ptr, key)
        setattr(ptr, keys[-1], torch.nn.Parameter(tensor))
            
def load_sharded_model(model_dir: str, model_class):
    # Step 1: Load the config
    config = AutoConfig.from_pretrained(model_dir)
    
    # Step 2: Create model on meta device (no memory used)
    with init_empty_weights():
        # Use the provided model_class to instantiate the correct architecture
        model = model_class.from_config(config,
            attn_implementation="eager",
        )
    
    # Verify it's on meta device
    print(f"Model device: {next(model.parameters()).device}")  # Should print "meta"
    
    # Step 3: Load weights directly to GPU
    loader = ShardedSafeTensorsLoader(model_dir)
    model = loader.load_to_gpu(model)
    
    # Now model is on GPU
    print(f"Model device after loading: {next(model.parameters()).device}")  # Should print "cuda:0"
    
    return model
