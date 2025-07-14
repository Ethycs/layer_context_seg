# Layered Context Graph System - Checkpoint

## Current Status (July 13, 2025)

Enhancing the Layered Context Graph system to leverage attention mechanisms for processing long-context documents, segmenting into meaningful partitions, extracting attention patterns, building knowledge graphs, and saving reorganized output.

## Completed Work

- `attention_extractor.py` with basic attention pattern extraction
- Support for Hugging Face transformers and Ollama GGUF models
- Head specialization discovery and natural language rule application
- Boundary detection and semantic clustering
- NVIDIA GPU acceleration setup in dev container

## Next Steps

1. **GGUF Loading Into PyTorch** - Enhance `ollama_extractor.py` to load GGUF files into PyTorch tensors
2. **Percolation Context Windows** - Implement 15-30% overlap for optimal connectivity in `context_window.py`
3. **Natural Language Rules** - Create instruction seeder module for rule-based attention biasing
4. **Knowledge Graph Construction** - Build graph structure with node classification (KEEP/DELETE/TRACK)
5. **Document Reconstruction** - Implement reverse reconstruction with programmatic attention

## Key Architecture Insights

- **Attention-Driven Segmentation**: Use attention patterns instead of hand-crafted rules
- **Head Specialization**: Different heads for different content types
- **Percolation Theory**: 15-30% overlap optimal for window connectivity
- **Natural Language Rules**: Describe segmentation rules in natural language to bias attention heads
- **Interpretable System**: Visualize which heads make decisions

## GPU Status
NVIDIA CUDA libraries available for PyTorch tensor acceleration.
   - Implement reverse reconstruction with programmatic attention
   - Ensure preservation of code samples and technical content
   - Support XML-based notebook output formatting

5. **Test and Validate**
   - Test with various document types (conversations, technical docs, code)
   - Compare performance of different model types
   - Validate attention mechanism effects on partitioning quality

## Technical Core Components

- **Head Specialization**: Automatically finding what each head is good at
- **Dynamic Head Selection**: Choosing different heads for different text types
- **Programmable Neural Reorganizer**: Describing segmentation and reorganization rules in natural language
- **Rule-Guided Attention**: Using natural language rules to shape how attention heads behave

## Architecture Insights

The system leverages a key insight: instead of hand-crafted rules, we can let the model learn what makes good boundaries/relationships using the same attention mechanism for both segmentation and reorganization. This makes the system interpretable (visualizing which heads make decisions) and adaptive (different heads for different content).

## GPU Support

The system runs in a dev container with NVIDIA CUDA libraries pre-installed, allowing for GPU acceleration of transformer models.
