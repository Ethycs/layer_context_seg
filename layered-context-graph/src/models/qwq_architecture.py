#!/usr/bin/env python3
"""
Standard PyTorch Model Architecture for QwQ-32B
===============================================
This module now uses the standard Hugging Face Qwen2 implementation,
ensuring stability and correctness. The custom architecture has been removed.
"""

from transformers import Qwen2Config, Qwen2ForCausalLM

# The QwQConfig is now a direct alias for the standard Qwen2Config,
# as the model loading process will dynamically create the correct
# configuration from the GGUF file's metadata.
QwQConfig = Qwen2Config

# The QwQForCausalLM is now a direct alias for the standard Qwen2ForCausalLM.
# This eliminates all custom, error-prone forward pass logic.
QwQForCausalLM = Qwen2ForCausalLM
