"""
Consolidated Configuration for Layered Context Graph Master Processor
====================================================================

This file replaces the various config files with a unified configuration system.
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "layered-context-graph" / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Core processing configurations
PROCESSING_MODES = {
    'single-pass': {
        'description': 'Fast single-pass processing with basic attention extraction',
        'suitable_for': ['quick analysis', 'simple documents', 'testing'],
        'components': ['ContextWindow', 'AttentionExtractor', 'PartitionManager', 'GraphReassembler']
    },
    'multi-round': {
        'description': 'Deep multi-round annotation with layered analysis',
        'suitable_for': ['complex documents', 'academic papers', 'detailed analysis'],
        'components': ['AnnotatedKnowledgeGraph', 'MultiRoundProcessor', 'CrossLayerSynthesis']
    },
    'language-guided': {
        'description': 'Natural language rule-guided processing',
        'suitable_for': ['custom workflows', 'domain-specific processing', 'interactive use'],
        'components': ['LanguageGuidedProcessor', 'RuleInterpreter']
    }
}

# Model configurations
MODEL_CONFIGS = {
    'transformer': {
        'default_model': 'distilbert-base-uncased',
        'alternatives': [
            'bert-base-uncased',
            'roberta-base',
            'sentence-transformers/all-MiniLM-L6-v2'
        ],
        'config': {
            'attention_heads': 12,
            'embedding_dim': 768,
            'max_sequence_length': 512
        }
    },
    'ollama': {
        'default_model': 'qwq',
        'gguf_path': PROJECT_ROOT / 'qwq.gguf',
        'alternatives': [
            'llama2',
            'codellama',
            'mistral'
        ],
        'config': {
            'static_analysis': True,
            'max_tokens': 4096,
            'temperature': 0.7
        }
    }
}

# Multi-round annotation configuration
MULTI_ROUND_CONFIG = {
    'analysis_rounds': {
        'syntactic': {
            'model': 'spacy_en_core_web_lg',
            'features': ['pos_tags', 'dependencies', 'syntax_patterns', 'linguistic_features'],
            'weight': 0.2,
            'description': 'Grammatical structure and linguistic patterns'
        },
        'semantic': {
            'model': 'sentence-transformers/all-MiniLM-L6-v2',
            'features': ['topics', 'concepts', 'semantic_roles', 'domain_classification'],
            'weight': 0.5,
            'description': 'Meaning and conceptual relationships'
        },
        'pragmatic': {
            'model': 'microsoft/DialoGPT-medium',
            'features': ['intent', 'discourse', 'rhetoric', 'contextual_importance'],
            'weight': 0.3,
            'description': 'Context, intent, and communicative purpose'
        }
    },
    'synthesis': {
        'cross_layer_analysis': True,
        'confidence_threshold': 0.7,
        'layer_weight_normalization': True,
        'pattern_detection': True
    }
}

# Processing settings
PROCESSING_SETTINGS = {
    # Text preprocessing
    'window_size': 2000,  # Default semantic window size
    'overlap_ratio': 0.1,  # Overlap between windows
    'min_chunk_size': 100,  # Minimum chunk size in characters
    
    # Fluff removal
    'fluff_removal': True,
    'preserve_code_blocks': True,
    'preserve_technical_terms': True,
    'aggressive_deduplication': False,
    
    # Graph construction
    'single_pass_processing': True,
    'preserve_original_content': True,
    'conservative_deduplication': True,
    'similarity_threshold': 0.95,
    
    # Reconstruction
    'use_document_scaffold': True,
    'scaffold_alignment_threshold': 0.8,
    'maintain_document_flow': True
}

# Fluff removal patterns (consolidated from various config files)
FLUFF_PATTERNS = {
    'filler_words': [
        r'\b(um|uh|like|you know|basically|actually|literally)\b',
        r'\b(obviously|clearly|definitely|probably|maybe|perhaps)\b',
        r'\b(kinda|sorta|yeah|yes|no|sure|well|so|anyway)\b',
        r'\b(alright|ok|okay|right)\b'
    ],
    'repetitive_patterns': [
        r'\b(.+?)\b\s+\1\b',  # Repeated words
        r'(.{10,}?)\s+\1',   # Repeated phrases
    ],
    'whitespace_cleanup': [
        r'\s+',      # Multiple whitespace
        r'\n{3,}',   # Multiple newlines
        r'\.{2,}',   # Multiple dots
        r'\?{2,}',   # Multiple question marks
        r'!{2,}',    # Multiple exclamation marks
    ]
}

# Code preservation patterns
CODE_PRESERVATION = {
    'markers': [
        '```', '`', 'def ', 'class ', 'import ', 'from ', 
        '#!/', '</', '/>', '{}', '[]', '()', '=>', '->'
    ],
    'language_indicators': [
        'python', 'javascript', 'java', 'cpp', 'html', 'css',
        'sql', 'bash', 'shell', 'json', 'xml', 'yaml'
    ],
    'preserve_indentation': True,
    'preserve_comments': True
}

# Predefined rule sets for different use cases
RULE_SETS = {
    'academic_paper': {
        'segmentation': 'Split at section headers, mathematical formulations, and major concept introductions',
        'reorganization': 'Group by: abstract concepts, mathematical foundations, experimental results, conclusions',
        'preserve_patterns': ['equations', 'citations', 'figures', 'tables']
    },
    'meeting_transcript': {
        'segmentation': 'Split at speaker changes, topic shifts, and decision points',
        'reorganization': 'Group by: agenda items, decisions made, action items, follow-up tasks',
        'preserve_patterns': ['speaker_labels', 'timestamps', 'action_items']
    },
    'technical_documentation': {
        'segmentation': 'Split at major conceptual shifts, code blocks, and API descriptions',
        'reorganization': 'Group by: theoretical foundations, implementation details, code examples, usage patterns',
        'preserve_patterns': ['code_blocks', 'api_signatures', 'examples']
    },
    'conversation_analysis': {
        'segmentation': 'Split at natural topic boundaries, maintaining conversational flow',
        'reorganization': 'Group by: main themes, supporting details, examples, conclusions',
        'preserve_patterns': ['dialogue_structure', 'turn_taking', 'context_shifts']
    },
    'code_documentation': {
        'segmentation': 'Split at function definitions, class boundaries, and major algorithmic sections',
        'reorganization': 'Group by: interfaces, implementations, utilities, tests',
        'preserve_patterns': ['function_signatures', 'docstrings', 'comments', 'imports']
    }
}

# Output formatting options
OUTPUT_FORMATS = {
    'json': {
        'extension': '.json',
        'description': 'Structured JSON output with full metadata',
        'include_metadata': True,
        'human_readable': False
    },
    'markdown': {
        'extension': '.md',
        'description': 'Human-readable Markdown with preserved formatting',
        'include_metadata': True,
        'human_readable': True
    },
    'txt': {
        'extension': '.txt',
        'description': 'Plain text output',
        'include_metadata': False,
        'human_readable': True
    },
    'html': {
        'extension': '.html',
        'description': 'HTML output with interactive elements',
        'include_metadata': True,
        'human_readable': True
    }
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_logging': True,
    'console_logging': True,
    'log_dir': RESULTS_DIR / 'logs'
}

# Performance and resource settings
PERFORMANCE_CONFIG = {
    'max_memory_mb': 4096,  # Maximum memory usage in MB
    'max_processing_time_seconds': 300,  # Maximum processing time
    'parallel_processing': True,
    'max_workers': 4,
    'batch_size': 32,
    'use_gpu': True,  # Use GPU if available
    'cache_results': True,
    'cache_dir': RESULTS_DIR / 'cache'
}

# Demo content configurations
DEMO_CONFIGS = {
    'transcript': {
        'type': 'meeting_transcript',
        'rules': 'meeting_transcript',
        'expected_features': ['speaker_changes', 'topic_shifts', 'technical_discussion']
    },
    'technical': {
        'type': 'technical_documentation',
        'rules': 'technical_documentation',
        'expected_features': ['code_blocks', 'concepts', 'implementations']
    },
    'simple': {
        'type': 'conversation_analysis',
        'rules': 'conversation_analysis',
        'expected_features': ['paragraphs', 'simple_concepts']
    }
}

# Validation and quality control
QUALITY_CONTROL = {
    'min_confidence_threshold': 0.6,
    'require_coherent_output': True,
    'validate_code_preservation': True,
    'check_information_loss': True,
    'similarity_check_threshold': 0.8
}

def get_config(mode: str = 'single-pass', 
               model_type: str = 'transformer', 
               custom_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get a complete configuration for the specified mode and model type.
    
    Args:
        mode: Processing mode ('single-pass', 'multi-round', 'language-guided')
        model_type: Model type ('transformer', 'ollama')
        custom_overrides: Dictionary of custom settings to override defaults
    
    Returns:
        Complete configuration dictionary
    """
    
    config = {
        'mode': mode,
        'model_type': model_type,
        'model_config': MODEL_CONFIGS[model_type],
        'processing_settings': PROCESSING_SETTINGS.copy(),
        'fluff_patterns': FLUFF_PATTERNS,
        'code_preservation': CODE_PRESERVATION,
        'output_formats': OUTPUT_FORMATS,
        'logging': LOGGING_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'quality_control': QUALITY_CONTROL,
        'paths': {
            'project_root': PROJECT_ROOT,
            'src_dir': SRC_DIR,
            'results_dir': RESULTS_DIR,
            'models_dir': MODELS_DIR
        }
    }
    
    # Add mode-specific configuration
    if mode == 'multi-round':
        config['multi_round'] = MULTI_ROUND_CONFIG
    
    # Apply custom overrides
    if custom_overrides:
        config.update(custom_overrides)
    
    return config

def get_rule_set(rule_name: str) -> Dict[str, str]:
    """Get a predefined rule set by name"""
    return RULE_SETS.get(rule_name, RULE_SETS['technical_documentation'])

def list_available_models(model_type: str = None) -> List[str]:
    """List available models for a given type or all types"""
    if model_type:
        return [MODEL_CONFIGS[model_type]['default_model']] + MODEL_CONFIGS[model_type]['alternatives']
    else:
        models = []
        for mtype, config in MODEL_CONFIGS.items():
            models.extend([config['default_model']] + config['alternatives'])
        return models

# Export main configuration function
__all__ = ['get_config', 'get_rule_set', 'list_available_models', 'RULE_SETS', 'DEMO_CONFIGS']
