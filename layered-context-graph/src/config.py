"""Configuration settings for Layered Context Graph SystemSimplified, source-first, scaffold-guided approach"""import osfrom pathlib import Path# Core system configurationDEFAULT_CONFIG = {    # Source processing    'window_size': 8192,  # Words per semantic chunk    'fluff_removal': True,  # Enable source-level cleanup    'preserve_code_blocks': True,  # Protect technical content        # Graph construction      'single_pass_processing': True,  # No redundant partitioning    'preserve_original_content': True,  # Never modify source text    'conservative_deduplication': True,  # 95% similarity threshold        # Reconstruction    'use_document_scaffold': True,  # Use original as template    'scaffold_alignment_threshold': 0.8,  # Structure preservation        # Model settings    'model_type': 'transformer',  # or 'ollama'    'model_name': 'distilbert-base-uncased',    'attention_heads': 12,    'embedding_dim': 768}# Ollama-specific configurationOLLAMA_CONFIG = {    **DEFAULT_CONFIG,    'model_type': 'ollama',    'model_name': 'qwq',    'gguf_path': '/workspace/qwq.gguf',    'static_analysis': True,  # Analyze weights without inference}# Fluff removal patternsFLUFF_PATTERNS = [    # Filler words    r'\b(um|uh|like|you know|basically|actually|literally|obviously|clearly|definitely|probably|maybe|perhaps|kinda|sorta)\b',    # Conversation markers      r'\b(well|so|anyway|alright|ok|okay|right|yeah|yes|no|sure)\b',    # Whitespace cleanup    r'\s+',  # Multiple whitespace    r'\n{3,}',  # Multiple newlines    r'\.{2,}',  # Multiple dots    r'\?{2,}',  # Multiple question marks    r'!{2,}',  # Multiple exclamation marks
]

# Code block markers to preserve
CODE_MARKERS = ['```', '`', 'def ', 'class ', 'import ', 'from ', '#!/', '</', '/>', '{}', '[]', '()']

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = Path("/workspace/results")
RESULTS_DIR.mkdir(exist_ok=True)

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}
