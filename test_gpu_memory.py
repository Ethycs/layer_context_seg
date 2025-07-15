#\!/usr/bin/env python3
"""Test GPU memory management"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "layered-context-graph" / "src"))

from graph.torch_spectral_clustering import TorchSpectralClustering
from graph.torch_attention_graph_builder import TorchAttentionGraphBuilder

def test_gpu_memory():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.4f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.4f} GB")
        
        # Test spectral clustering
        print("\n--- Testing TorchSpectralClustering ---")
        clustering = TorchSpectralClustering(device=device)
        
        # Create test data
        seq_len = 512
        n_heads = 40
        attention = torch.rand(n_heads, seq_len, seq_len, device=device)
        
        print(f"After creating attention tensor:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.4f} GB")
        
        # Run clustering
        clusters, metadata = clustering(attention.mean(dim=0), num_clusters=8)
        
        print(f"After clustering:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.4f} GB")
        
        # Test graph builder
        print("\n--- Testing TorchAttentionGraphBuilder ---")
        graph_builder = TorchAttentionGraphBuilder(device=device)
        
        # Create dummy text segments
        segments = [f"Segment {i}" for i in range(8)]
        cluster_labels = torch.randint(0, 4, (8,), device=device)
        
        # Graph builder expects different inputs - just test memory allocation
        # graph = graph_builder(segments, cluster_labels, attention[:, :8, :8])
        
        print(f"After graph builder init:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.4f} GB")
        
        # Clear cache
        torch.cuda.empty_cache()
        print(f"\nAfter clearing cache:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.4f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.4f} GB")

if __name__ == "__main__":
    test_gpu_memory()
