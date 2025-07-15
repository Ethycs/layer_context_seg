#!/usr/bin/env python3
"""
Test Real Attention Extraction from GGUF
========================================
Validates that real attention patterns produce meaningful spectral gaps.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "layered-context-graph" / "src"))

from models.gguf_attention_extractor import GGUFAttentionExtractor
from models.ollama_extractor import OllamaModelExtractor


def test_attention_extraction():
    """Test that we can extract real attention patterns."""
    
    print("=" * 60)
    print("Testing Real Attention Extraction")
    print("=" * 60)
    
    # Test text
    test_text = """
    Speaker A: We should implement microservices for better scalability.
    Speaker B: But what about the added complexity of managing multiple services?
    Speaker A: True, there's operational complexity, but the benefits outweigh it.
    Speaker B: Earlier you advocated for simplicity. This seems contradictory.
    """
    
    # Try to find GGUF model
    model_paths = [
        "qwq.gguf",
        "/home/user/.ollama/models/blobs/sha256-c89a3a433e94f68f4259e70e7c97de5e68ff96e7f6c17ec7ad8c53f3e2c88c4a",
        "/root/.ollama/models/blobs/sha256-c89a3a433e94f68f4259e70e7c97de5e68ff96e7f6c17ec7ad8c53f3e2c88c4a",
        "./qwq-32b-preview.gguf"
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if not model_path:
        print("❌ No GGUF model found. Please ensure QwQ model is available.")
        print("   Checked paths:")
        for path in model_paths:
            print(f"   - {path}")
        return
    
    print(f"✅ Found model at: {model_path}")
    
    # Test 1: Direct GGUF attention extraction
    print("\n1. Testing GGUFAttentionExtractor...")
    print("❌ Skipping - GGUF file contains quantized weights that need proper dequantization")
    print("   The current implementation expects full precision weights.")
    
    # Test 2: OllamaModelExtractor with real attention
    print("\n2. Testing OllamaModelExtractor.get_attention_patterns()...")
    try:
        ollama_extractor = OllamaModelExtractor(model_path, device=device)
        patterns = ollama_extractor.get_attention_patterns(test_text)
        
        if patterns:
            print(f"✅ Extracted {len(patterns)} attention patterns via Ollama")
            
            # Analyze one pattern in detail
            if 0 in patterns:
                attention = patterns[0]
                print(f"\n   Analyzing Layer 0 attention:")
                print(f"   - Shape: {attention.shape}")
                print(f"   - Device: {attention.device}")
                
                # Compute simple spectral gap
                if attention.dim() >= 2:
                    # Move to device if not already there
                    if attention.device != device:
                        attention = attention.to(device)
                    
                    # Average over heads if multi-head
                    if attention.dim() == 3:
                        avg_attention = attention.mean(dim=0)
                    else:
                        avg_attention = attention
                    
                    # Make symmetric
                    symmetric = (avg_attention + avg_attention.t()) / 2
                    
                    # Compute Laplacian
                    D = torch.diag(symmetric.sum(dim=1))
                    L = D - symmetric
                    
                    # Get eigenvalues
                    eigenvalues = torch.linalg.eigvalsh(L)
                    spectral_gap = (eigenvalues[1] - eigenvalues[0]).item()
                    
                    print(f"\n   Spectral Analysis:")
                    print(f"   - Smallest eigenvalues: {eigenvalues[:3].cpu().numpy()}")
                    print(f"   - Spectral gap: {spectral_gap:.6f}")
                    
                    if spectral_gap > 0.0001:
                        print("   ✅ Non-zero spectral gap - Real attention patterns!")
                    else:
                        print("   ❌ Zero/tiny spectral gap - Likely synthetic patterns")
        else:
            print("❌ No patterns returned")
    except Exception as e:
        print(f"❌ OllamaModelExtractor failed: {e}")
    
    # Test 3: Compare real vs synthetic
    print("\n3. Comparing Real vs Synthetic Attention...")
    
    # Synthetic pattern (create on GPU)
    seq_len = 10
    n_heads = 8
    synthetic = torch.rand(n_heads, seq_len, seq_len, device=device)
    
    # Make symmetric and compute spectral gap
    synthetic_avg = synthetic.mean(dim=0)
    synthetic_sym = (synthetic_avg + synthetic_avg.t()) / 2
    D_syn = torch.diag(synthetic_sym.sum(dim=1))
    L_syn = D_syn - synthetic_sym
    eigen_syn = torch.linalg.eigvalsh(L_syn)
    gap_syn = (eigen_syn[1] - eigen_syn[0]).item()
    
    print(f"\n   Synthetic attention spectral gap: {gap_syn:.6f}")
    print("   (Should be very small for random patterns)")
    
    # Visual comparison if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Plot synthetic
        ax = axes[0]
        im = ax.imshow(synthetic_avg.cpu().numpy(), cmap='hot')
        ax.set_title('Synthetic Attention (Random)')
        ax.set_xlabel('Position')
        ax.set_ylabel('Position')
        plt.colorbar(im, ax=ax)
        
        # Plot spectral gaps
        ax = axes[1]
        gaps = ['Synthetic', 'Real (if available)']
        values = [gap_syn, spectral_gap if 'spectral_gap' in locals() else 0]
        colors = ['red', 'green']
        bars = ax.bar(gaps, values, color=colors)
        ax.set_ylabel('Spectral Gap')
        ax.set_title('Spectral Gap Comparison')
        ax.set_ylim(0, max(0.001, max(values) * 1.2))
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00001,
                   f'{val:.6f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('attention_comparison.png', dpi=150)
        print("\n   Saved visualization to attention_comparison.png")
    except ImportError:
        pass
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Real attention patterns should show:")
    print("- Non-zero spectral gaps (typically > 0.0001)")
    print("- Structured patterns (not random noise)")
    print("- Higher variance than synthetic patterns")
    print("- Meaningful attention to related tokens")


if __name__ == "__main__":
    test_attention_extraction()