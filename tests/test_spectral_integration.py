#!/usr/bin/env python3
"""
Test script for spectral processing integration
"""

import torch
from master_processor import FullMasterProcessor
from master_config import get_config
from load_demo_content import get_demo_content

def test_spectral_integration():
    """Test the spectral-hybrid processing mode"""
    
    print("Testing Spectral-Hybrid Processing Integration")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ No GPU available - will use CPU")
    
    # Get configuration for spectral-hybrid mode
    config = get_config(mode='spectral-hybrid', model_type='ollama')
    config['use_spectral'] = True
    
    # Create processor
    try:
        processor = FullMasterProcessor(config)
        print("✅ FullMasterProcessor initialized successfully")
        
        if processor.spectral_processor:
            print("✅ TorchSpectralProcessor is available")
        else:
            print("❌ TorchSpectralProcessor not available")
            
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return
    
    # Test with conversation content
    conversation = get_demo_content('conversation')
    
    print("\nProcessing conversation with spectral-hybrid mode...")
    
    try:
        # Test spectral processing
        results = processor.process_text(conversation)
        
        print(f"\n✅ Processing completed successfully!")
        print(f"Mode: {results['mode']}")
        print(f"Nodes created: {results['nodes']}")
        print(f"Edges created: {results['edges']}")
        print(f"Processing time: {results['processing_time']:.2f}s")
        
        if 'spectral_metadata' in results.get('output', {}):
            meta = results['output']['spectral_metadata']
            print(f"\nSpectral Metadata:")
            print(f"- Eigenvalues: {meta.get('eigenvalues', 'N/A')}")
            print(f"- Clusters: {meta.get('num_clusters', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test conversation-specific processing with spectral option
    print("\n" + "="*60)
    print("Testing conversation processing with spectral option...")
    
    try:
        conv_results = processor.process_conversation(
            conversation, 
            mode='speaker',
            use_spectral=True
        )
        
        if 'graph' in conv_results:
            print(f"✅ Conversation processing with spectral completed!")
            print(f"Segments: {conv_results.get('segments', 'N/A')}")
            print(f"Speakers: {conv_results.get('speakers', 'N/A')}")
        else:
            print("✅ Standard conversation processing completed")
            
    except Exception as e:
        print(f"❌ Conversation processing failed: {e}")
    
    print("\n" + "="*60)
    print("Integration test complete!")


if __name__ == "__main__":
    test_spectral_integration()