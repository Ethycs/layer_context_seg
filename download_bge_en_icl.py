import requests
from tqdm import tqdm
import os
from pathlib import Path

def download_bge_en_icl():
    """
    Downloads the BAAI/bge-en-icl model files from Hugging Face.
    """
    # --- Configuration ---
    model_repo = "BAAI/bge-en-icl"
    output_dir_name = "bge-en-icl"
    
    # Files to download (excluding .gitattributes and README.md)
    files_to_download = [
        "added_tokens.json",
        "config.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "model.safetensors.index.json",
        "modules.json",
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "1_Pooling/config.json"
    ]
    # ---------------------

    base_url = f"https://huggingface.co/{model_repo}/resolve/main/"
    output_dir = Path(f"./{output_dir_name}")
    output_dir.mkdir(exist_ok=True)
    
    print(f"--- Starting Download for {model_repo} ---")
    print(f"Files will be saved in: {output_dir.resolve()}")
    
    for filename in files_to_download:
        download_url = f"{base_url}{filename}?download=true"
        output_path = output_dir / filename
        
        # Create subdirectory if needed
        if "/" in filename:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.exists():
            print(f"File '{filename}' already exists. Skipping.")
            continue
            
        print(f"Downloading '{filename}'...")
        
        try:
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                with open(output_path, 'wb') as f, tqdm(
                    total=total_size, unit='iB', unit_scale=True, desc=filename
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            print(f"Successfully downloaded '{filename}'")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download '{filename}'. Error: {e}")
            if output_path.exists():
                os.remove(output_path) # Clean up partial download
            print("Stopping download process due to error.")
            break
    
    print("\n--- Download Complete ---")

if __name__ == "__main__":
    # Ensure you have the necessary libraries: pip install requests tqdm
    download_bge_en_icl()