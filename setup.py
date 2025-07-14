#!/usr/bin/env python3
"""
Setup Script for Layered Context Graph Master Processor
======================================================

This script helps set up the environment and verify that all components work correctly.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def download_spacy_model():
    """Download spaCy language model"""
    print("\n🔤 Downloading spaCy English model...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "spacy", "download", "en_core_web_lg"
        ], check=True)
        print("✅ spaCy model downloaded successfully")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Failed to download spaCy model (optional)")
        return False

def check_optional_components():
    """Check if optional components are available"""
    print("\n🔍 Checking optional components...")
    
    # Check for Ollama
    try:
        import ollama
        print("✅ Ollama integration available")
    except ImportError:
        print("⚠️  Ollama not available (optional)")
    
    # Check for GPU support
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ GPU support available")
        else:
            print("ℹ️  GPU not available, using CPU")
    except ImportError:
        print("⚠️  PyTorch not available")

def verify_installation():
    """Verify that the installation works"""
    print("\n🧪 Verifying installation...")
    
    try:
        # Try importing the master processor
        sys.path.append(str(Path(__file__).parent))
        from master_config import get_config, get_rule_set
        
        # Test configuration loading
        config = get_config('single-pass', 'transformer')
        rules = get_rule_set('technical_documentation')
        
        print("✅ Configuration system working")
        print("✅ Installation verification successful")
        return True
        
    except Exception as e:
        print(f"❌ Installation verification failed: {e}")
        return False

def create_demo_files():
    """Create demo files for testing"""
    print("\n📄 Creating demo files...")
    
    demo_dir = Path(__file__).parent / "demo"
    demo_dir.mkdir(exist_ok=True)
    
    # Create a simple demo file
    demo_content = """
# Demo Document for Layered Context Graph Processing

This is a demonstration document that shows different types of content
that the layered context graph system can process.

## Technical Section

Here's some technical content with code:

```python
def process_text(text):
    # This function processes text using layered context graphs
    chunks = create_semantic_chunks(text)
    graph = build_knowledge_graph(chunks)
    return reassemble_from_graph(graph)
```

## Discussion Section

This section contains conversational content that might appear in
meeting transcripts or interview data.

Speaker A: "What's the best approach for handling long documents?"

Speaker B: "We should use a layered context approach where we segment
the text into semantic chunks and then connect them using a knowledge graph."

## Mathematical Section

The system is based on percolation theory and graph algorithms:

- Connectivity threshold: τ = 0.593
- Percolation probability: p = 1 - exp(-βτ)
- Graph density: ρ = |E| / |V|²

## Conclusion

This demonstrates various content types that the system can handle
effectively while preserving structure and meaning.
"""
    
    demo_file = demo_dir / "sample_document.txt"
    with open(demo_file, 'w') as f:
        f.write(demo_content)
    
    print(f"✅ Demo file created: {demo_file}")
    return demo_file

def print_usage_examples():
    """Print usage examples"""
    print("\n📚 Usage Examples:")
    print("=" * 50)
    
    examples = [
        "# Single-pass processing with demo content",
        "python master_processor.py --mode single-pass --demo technical",
        "",
        "# Multi-round analysis with custom rules",
        "python master_processor.py --mode multi-round --rules academic_paper --verbose",
        "",
        "# Language-guided processing with transcript",
        "python master_processor.py --mode language-guided --demo transcript --preserve-code",
        "",
        "# Process custom file",
        "python master_processor.py --input demo/sample_document.txt --mode multi-round",
        "",
        "# Use Ollama model (if available)",
        "python master_processor.py --model-type ollama --model-name qwq --demo technical",
    ]
    
    for example in examples:
        print(example)

def main():
    """Main setup function"""
    print("🚀 Layered Context Graph Master Processor Setup")
    print("=" * 50)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Install dependencies
    if success and not install_dependencies():
        success = False
    
    # Download optional models
    download_spacy_model()  # Optional, don't fail if this doesn't work
    
    # Check optional components
    check_optional_components()
    
    # Verify installation
    if success and not verify_installation():
        success = False
    
    # Create demo files
    if success:
        demo_file = create_demo_files()
    
    # Print results
    print("\n" + "=" * 50)
    if success:
        print("🎉 Setup completed successfully!")
        print("\nYou can now use the master processor with:")
        print("python master_processor.py --help")
        print_usage_examples()
    else:
        print("❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
