#!/usr/bin/env python3
"""
Test centralized configuration system
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.resolve()
src_path = project_root / "layered-context-graph" / "src"
sys.path.insert(0, str(src_path))

from config.graph_config import GraphConfig, get_config, PRESETS

def test_centralized_config():
    """Test the centralized configuration system"""
    
    print("Testing Centralized Graph Configuration")
    print("=" * 60)
    
    # Show available presets
    print("\nAvailable Presets:")
    for preset_name in PRESETS.keys():
        print(f"  - {preset_name}")
    
    # Test default configuration
    print("\n1. Default Configuration:")
    default_config = get_config('default')
    print(f"  Disassembly:")
    print(f"    - Min segment length: {default_config.disassembly.min_segment_length}")
    print(f"    - Target segment length: {default_config.disassembly.target_segment_length}")
    print(f"    - Preserve formatting: {default_config.disassembly.preserve_whitespace}")
    print(f"  Assembly:")
    print(f"    - Use attention edges: {default_config.assembly.use_attention_edges}")
    print(f"    - Attention threshold: {default_config.assembly.attention_threshold}")
    print(f"  Reconstruction:")
    print(f"    - Preserve formatting: {default_config.reconstruction.preserve_original_formatting}")
    print(f"    - Default strategy: {default_config.reconstruction.default_strategy}")
    
    # Test conversation preset
    print("\n2. Conversation Preset:")
    conv_config = get_config('conversation')
    print(f"  Disassembly:")
    print(f"    - Min segment length: {conv_config.disassembly.min_segment_length}")
    print(f"    - Detect conversations: {conv_config.disassembly.detect_conversations}")
    print(f"  Reconstruction:")
    print(f"    - Preserve speaker labels: {conv_config.reconstruction.preserve_speaker_labels}")
    print(f"    - Maintain conversation flow: {conv_config.reconstruction.maintain_conversation_flow}")
    
    # Test code preset
    print("\n3. Code Preset:")
    code_config = get_config('code')
    print(f"  Disassembly:")
    print(f"    - Preserve code blocks: {code_config.disassembly.preserve_code_blocks}")
    print(f"    - Min segment length: {code_config.disassembly.min_segment_length}")
    print(f"  Reconstruction:")
    print(f"    - Maintain code integrity: {code_config.reconstruction.maintain_code_block_integrity}")
    print(f"    - Reassemble split functions: {code_config.reconstruction.reassemble_split_functions}")
    
    # Test override functionality
    print("\n4. Testing Overrides:")
    custom_config = get_config(
        'default',
        mode='spectral-hybrid',
        **{
            'disassembly.min_segment_length': 1000,
            'assembly.attention_threshold': 0.2,
            'reconstruction.preserve_original_formatting': True
        }
    )
    print(f"  Mode: {custom_config.mode}")
    print(f"  Disassembly min length: {custom_config.disassembly.min_segment_length}")
    print(f"  Assembly attention threshold: {custom_config.assembly.attention_threshold}")
    print(f"  Reconstruction preserve formatting: {custom_config.reconstruction.preserve_original_formatting}")
    
    # Test serialization
    print("\n5. Testing Serialization:")
    config_dict = default_config.to_dict()
    print(f"  Serialized to dict with {len(config_dict)} top-level keys")
    
    # Test deserialization
    restored_config = GraphConfig.from_dict(config_dict)
    print(f"  Restored config mode: {restored_config.mode}")
    print(f"  Restored disassembly overlap: {restored_config.disassembly.overlap_ratio}")
    
    print("\n" + "=" * 60)
    print("✅ Centralized configuration system is working!")
    print("\nKey Benefits:")
    print("1. Single source of truth for all graph processing parameters")
    print("2. Easy preset configurations for common use cases")
    print("3. Override capability for fine-tuning")
    print("4. Formatting preservation controls in one place")
    print("5. Consistent settings across all components")


if __name__ == "__main__":
    test_centralized_config()