#!/usr/bin/env python3
"""
Centralized Configuration for Graph Assembly/Disassembly and Document Construction
==================================================================================
Single source of truth for all graph processing parameters.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DisassemblyConfig:
    """Configuration for document disassembly (Tape → Graph)"""
    
    # Segmentation parameters
    min_segment_length: int = 800
    target_segment_length: int = 1500  
    max_segment_length: int = 3000
    
    # Overlap for percolation theory (15-30% is optimal)
    overlap_ratio: float = 0.2
    
    # Preserve formatting
    preserve_whitespace: bool = True
    preserve_code_blocks: bool = True
    preserve_markdown: bool = True
    preserve_line_breaks: bool = True
    
    # Boundary detection
    respect_sentence_boundaries: bool = True
    respect_paragraph_boundaries: bool = True
    respect_section_boundaries: bool = True
    
    # Special handling
    detect_conversations: bool = True
    detect_code_sections: bool = True
    detect_lists: bool = True
    
    # Attention-based segmentation
    use_attention_boundaries: bool = True
    attention_boundary_threshold: float = 0.3


@dataclass
class AssemblyConfig:
    """Configuration for graph assembly and edge creation"""
    
    # Edge detection methods
    use_attention_edges: bool = True
    use_percolation_edges: bool = True
    use_semantic_edges: bool = True
    
    # Attention-based edges
    attention_threshold: float = 0.15
    attention_aggregation: str = 'weighted'  # 'mean', 'max', 'weighted'
    
    # Percolation theory parameters
    min_overlap_ratio: float = 0.15
    max_overlap_ratio: float = 0.30
    percolation_weight_multiplier: float = 2.0
    
    # Semantic similarity
    semantic_threshold: float = 0.7
    use_embeddings: bool = False  # Future: use embeddings for similarity
    
    # Edge types to create
    create_sequential_edges: bool = True
    create_reference_edges: bool = True
    create_hierarchy_edges: bool = True
    create_conversation_edges: bool = True
    
    # Graph properties
    make_graph_symmetric: bool = True
    remove_self_loops: bool = True
    edge_weight_normalization: bool = True


@dataclass
class ReconstructionConfig:
    """Configuration for document reconstruction (Graph → Tape)"""
    
    # Formatting preservation
    preserve_original_formatting: bool = True
    preserve_indentation: bool = True
    preserve_empty_lines: bool = True
    maintain_code_block_integrity: bool = True
    
    # Reconstruction strategies
    default_strategy: str = 'hierarchical'  # 'linear', 'hierarchical', 'thematic'
    
    # Output formatting
    add_section_headers: bool = True
    add_transition_text: bool = False
    maintain_paragraph_spacing: bool = True
    
    # Special handling
    merge_short_segments: bool = True
    merge_threshold: int = 100  # characters
    
    # Conversation reconstruction
    preserve_speaker_labels: bool = True
    maintain_conversation_flow: bool = True
    
    # Code reconstruction
    reassemble_split_functions: bool = True
    maintain_import_order: bool = True
    preserve_docstrings: bool = True


@dataclass
class SpectralConfig:
    """Configuration for spectral processing"""
    
    # Spectral clustering
    use_spectral_clustering: bool = True
    normalized_laplacian: bool = True
    num_clusters: Optional[int] = None  # None = auto-detect
    
    # GPU acceleration
    force_gpu: bool = False
    device: Optional[str] = None  # None = auto-select
    
    # Linguistic programming
    use_linguistic_programming: bool = True
    instruction_density: float = 0.15
    
    # Hierarchical clustering
    max_hierarchy_depth: int = 3
    min_cluster_size: int = 2


@dataclass
class GraphConfig:
    """Master configuration for the entire graph processing pipeline"""
    
    # Sub-configurations
    disassembly: DisassemblyConfig = field(default_factory=DisassemblyConfig)
    assembly: AssemblyConfig = field(default_factory=AssemblyConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    
    # Global settings
    mode: str = 'single-pass'  # 'single-pass', 'multi-round', 'language-guided', 'spectral-hybrid'
    verbose: bool = False
    debug: bool = False
    
    # Model settings
    model_path: Optional[Path] = None
    use_qwq: bool = True
    
    # Performance settings
    cache_attention: bool = True
    batch_processing: bool = True
    max_batch_size: int = 8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'disassembly': {
                'min_segment_length': self.disassembly.min_segment_length,
                'target_segment_length': self.disassembly.target_segment_length,
                'max_segment_length': self.disassembly.max_segment_length,
                'overlap_ratio': self.disassembly.overlap_ratio,
                'preserve_whitespace': self.disassembly.preserve_whitespace,
                'preserve_code_blocks': self.disassembly.preserve_code_blocks,
                'preserve_markdown': self.disassembly.preserve_markdown,
                'preserve_line_breaks': self.disassembly.preserve_line_breaks,
                'respect_sentence_boundaries': self.disassembly.respect_sentence_boundaries,
                'respect_paragraph_boundaries': self.disassembly.respect_paragraph_boundaries,
                'respect_section_boundaries': self.disassembly.respect_section_boundaries,
                'detect_conversations': self.disassembly.detect_conversations,
                'detect_code_sections': self.disassembly.detect_code_sections,
                'detect_lists': self.disassembly.detect_lists,
                'use_attention_boundaries': self.disassembly.use_attention_boundaries,
                'attention_boundary_threshold': self.disassembly.attention_boundary_threshold,
            },
            'assembly': {
                'use_attention_edges': self.assembly.use_attention_edges,
                'use_percolation_edges': self.assembly.use_percolation_edges,
                'use_semantic_edges': self.assembly.use_semantic_edges,
                'attention_threshold': self.assembly.attention_threshold,
                'attention_aggregation': self.assembly.attention_aggregation,
                'min_overlap_ratio': self.assembly.min_overlap_ratio,
                'max_overlap_ratio': self.assembly.max_overlap_ratio,
                'percolation_weight_multiplier': self.assembly.percolation_weight_multiplier,
                'semantic_threshold': self.assembly.semantic_threshold,
                'use_embeddings': self.assembly.use_embeddings,
                'create_sequential_edges': self.assembly.create_sequential_edges,
                'create_reference_edges': self.assembly.create_reference_edges,
                'create_hierarchy_edges': self.assembly.create_hierarchy_edges,
                'create_conversation_edges': self.assembly.create_conversation_edges,
                'make_graph_symmetric': self.assembly.make_graph_symmetric,
                'remove_self_loops': self.assembly.remove_self_loops,
                'edge_weight_normalization': self.assembly.edge_weight_normalization,
            },
            'reconstruction': {
                'preserve_original_formatting': self.reconstruction.preserve_original_formatting,
                'preserve_indentation': self.reconstruction.preserve_indentation,
                'preserve_empty_lines': self.reconstruction.preserve_empty_lines,
                'maintain_code_block_integrity': self.reconstruction.maintain_code_block_integrity,
                'default_strategy': self.reconstruction.default_strategy,
                'add_section_headers': self.reconstruction.add_section_headers,
                'add_transition_text': self.reconstruction.add_transition_text,
                'maintain_paragraph_spacing': self.reconstruction.maintain_paragraph_spacing,
                'merge_short_segments': self.reconstruction.merge_short_segments,
                'merge_threshold': self.reconstruction.merge_threshold,
                'preserve_speaker_labels': self.reconstruction.preserve_speaker_labels,
                'maintain_conversation_flow': self.reconstruction.maintain_conversation_flow,
                'reassemble_split_functions': self.reconstruction.reassemble_split_functions,
                'maintain_import_order': self.reconstruction.maintain_import_order,
                'preserve_docstrings': self.reconstruction.preserve_docstrings,
            },
            'spectral': {
                'use_spectral_clustering': self.spectral.use_spectral_clustering,
                'normalized_laplacian': self.spectral.normalized_laplacian,
                'num_clusters': self.spectral.num_clusters,
                'force_gpu': self.spectral.force_gpu,
                'device': self.spectral.device,
                'use_linguistic_programming': self.spectral.use_linguistic_programming,
                'instruction_density': self.spectral.instruction_density,
                'max_hierarchy_depth': self.spectral.max_hierarchy_depth,
                'min_cluster_size': self.spectral.min_cluster_size,
            },
            'mode': self.mode,
            'verbose': self.verbose,
            'debug': self.debug,
            'model_path': str(self.model_path) if self.model_path else None,
            'use_qwq': self.use_qwq,
            'cache_attention': self.cache_attention,
            'batch_processing': self.batch_processing,
            'max_batch_size': self.max_batch_size,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GraphConfig':
        """Create configuration from dictionary"""
        config = cls()
        
        # Update disassembly config
        if 'disassembly' in config_dict:
            for key, value in config_dict['disassembly'].items():
                if hasattr(config.disassembly, key):
                    setattr(config.disassembly, key, value)
        
        # Update assembly config
        if 'assembly' in config_dict:
            for key, value in config_dict['assembly'].items():
                if hasattr(config.assembly, key):
                    setattr(config.assembly, key, value)
        
        # Update reconstruction config
        if 'reconstruction' in config_dict:
            for key, value in config_dict['reconstruction'].items():
                if hasattr(config.reconstruction, key):
                    setattr(config.reconstruction, key, value)
        
        # Update spectral config
        if 'spectral' in config_dict:
            for key, value in config_dict['spectral'].items():
                if hasattr(config.spectral, key):
                    setattr(config.spectral, key, value)
        
        # Update global settings
        for key in ['mode', 'verbose', 'debug', 'use_qwq', 'cache_attention', 
                    'batch_processing', 'max_batch_size']:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        if 'model_path' in config_dict and config_dict['model_path']:
            config.model_path = Path(config_dict['model_path'])
        
        return config


# Preset configurations for common use cases
PRESETS = {
    'default': GraphConfig(),
    
    'conversation': GraphConfig(
        disassembly=DisassemblyConfig(
            detect_conversations=True,
            respect_sentence_boundaries=True,
            min_segment_length=100,  # Shorter for conversation turns
            target_segment_length=500,
        ),
        assembly=AssemblyConfig(
            create_conversation_edges=True,
            use_attention_edges=True,
        ),
        reconstruction=ReconstructionConfig(
            preserve_speaker_labels=True,
            maintain_conversation_flow=True,
        )
    ),
    
    'code': GraphConfig(
        disassembly=DisassemblyConfig(
            detect_code_sections=True,
            preserve_code_blocks=True,
            respect_section_boundaries=True,
            min_segment_length=200,  # Don't split small functions
        ),
        assembly=AssemblyConfig(
            create_hierarchy_edges=True,
            use_semantic_edges=True,
        ),
        reconstruction=ReconstructionConfig(
            maintain_code_block_integrity=True,
            reassemble_split_functions=True,
            maintain_import_order=True,
            preserve_docstrings=True,
        )
    ),
    
    'research': GraphConfig(
        disassembly=DisassemblyConfig(
            use_attention_boundaries=True,
            attention_boundary_threshold=0.25,
            target_segment_length=2000,  # Larger segments for research
        ),
        assembly=AssemblyConfig(
            use_attention_edges=True,
            use_semantic_edges=True,
            create_reference_edges=True,
        ),
        reconstruction=ReconstructionConfig(
            default_strategy='thematic',
            add_section_headers=True,
        )
    ),
    
    'spectral': GraphConfig(
        mode='spectral-hybrid',
        spectral=SpectralConfig(
            use_spectral_clustering=True,
            use_linguistic_programming=True,
            force_gpu=True,
        ),
        assembly=AssemblyConfig(
            use_attention_edges=True,
            attention_aggregation='weighted',
        )
    ),
}


def get_config(preset: str = 'default', **overrides) -> GraphConfig:
    """
    Get a configuration preset with optional overrides.
    
    Args:
        preset: Name of the preset ('default', 'conversation', 'code', 'research', 'spectral')
        **overrides: Key-value pairs to override in the configuration
        
    Returns:
        GraphConfig instance
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
    
    # Start with preset
    config = PRESETS[preset]
    
    # Apply overrides
    if overrides:
        config_dict = config.to_dict()
        
        # Deep update for nested configs
        for key, value in overrides.items():
            if '.' in key:
                # Handle nested keys like 'disassembly.min_segment_length'
                parts = key.split('.')
                current = config_dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[key] = value
        
        config = GraphConfig.from_dict(config_dict)
    
    return config