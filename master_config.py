"""
Consolidated Configuration for Layered Context Graph Master Processor
====================================================================

This file replaces the various config files with a unified configuration system.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Project paths
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "layered-context-graph" / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# --- Merged GraphConfig Classes ---

@dataclass
class DisassemblyConfig:
    min_segment_length: int = 800
    target_segment_length: int = 1500
    max_segment_length: int = 3000
    overlap_ratio: float = 0.2
    preserve_whitespace: bool = True
    preserve_code_blocks: bool = True
    respect_sentence_boundaries: bool = True
    use_attention_boundaries: bool = True
    attention_boundary_threshold: float = 0.3

@dataclass
class AssemblyConfig:
    use_attention_edges: bool = True
    attention_threshold: float = 0.15
    semantic_threshold: float = 0.7
    create_sequential_edges: bool = True
    make_graph_symmetric: bool = True

@dataclass
class ReconstructionConfig:
    preserve_original_formatting: bool = True
    default_strategy: str = 'hierarchical'
    add_section_headers: bool = True
    preserve_speaker_labels: bool = True

@dataclass
class SpectralConfig:
    use_spectral_clustering: bool = True
    normalized_laplacian: bool = True
    num_clusters: Optional[int] = None
    force_gpu: bool = False

@dataclass
class GraphConfig:
    disassembly: DisassemblyConfig = field(default_factory=DisassemblyConfig)
    assembly: AssemblyConfig = field(default_factory=AssemblyConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)

# --- End of Merged GraphConfig Classes ---

# Processing settings (previously in a separate dict)
PROCESSING_SETTINGS = {
    'window_size': 2000,
    'overlap_ratio': 0.1,
    'min_chunk_size': 100,
    'fluff_removal': True,
    'preserve_code_blocks': True,
    'similarity_threshold': 0.95,
}

# Core processing configurations
PROCESSING_MODES = {
    'single-pass': {
        'description': 'Fast single-pass processing with basic attention extraction',
    },
    'rich': {
        'description': 'Rich processing with multi-round annotation and optional language guidance.',
    },
    'som-pipeline': {
        'description': 'Self-Organizing Map-based document generation',
    }
}

# Model configurations
MODEL_CONFIGS = {
    'transformer': {
        'default_model': 'distilbert-base-uncased',
    },
    'ollama': {
        'default_model': 'qwq',
        'gguf_path': PROJECT_ROOT / 'qwq.gguf',
    }
}

# Predefined rule sets for different use cases
RULE_SETS = {
    'academic_paper': {
        'segmentation': 'Split at section headers, mathematical formulations, and major concept introductions',
    },
    'meeting_transcript': {
        'segmentation': 'Split at speaker changes, topic shifts, and decision points',
    },
    'technical_documentation': {
        'segmentation': 'Split at major conceptual shifts, code blocks, and API descriptions',
    }
}

# Demo content configurations
DEMO_CONFIGS = {
    'transcript': {
        'type': 'meeting_transcript',
    },
    'technical': {
        'type': 'technical_documentation',
    },
    'simple': {
        'type': 'conversation_analysis',
    }
}

def get_config(mode: str = 'single-pass', 
               model_type: str = 'ollama', 
               custom_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get a complete configuration for the specified mode and model type.
    """
    
    config = {
        'mode': mode,
        'model_type': model_type,
        'model_config': MODEL_CONFIGS[model_type],
        'processing_settings': PROCESSING_SETTINGS,
        'paths': {
            'project_root': PROJECT_ROOT,
            'src_dir': SRC_DIR,
            'results_dir': RESULTS_DIR,
            'models_dir': MODELS_DIR
        },
        'graph_config': GraphConfig() # Add the merged graph config
    }
    
    # Apply custom overrides
    if custom_overrides:
        # Simple override for demonstration. A real implementation would be deeper.
        for key, value in custom_overrides.items():
            if key in config:
                config[key] = value
    
    return config

def get_rule_set(rule_name: str) -> Dict[str, str]:
    """Get a predefined rule set by name"""
    return RULE_SETS.get(rule_name, RULE_SETS['technical_documentation'])

# Export main configuration function
__all__ = ['get_config', 'get_rule_set', 'RULE_SETS', 'DEMO_CONFIGS', 'GraphConfig']
