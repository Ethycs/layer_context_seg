import requests
from tqdm import tqdm
import os
from pathlib import Path

def download_files():
    """
    Downloads the 14 shards of the QwQ-32B model from Hugging Face.
    """
    base_url = "https://huggingface.co/Qwen/QwQ-32B/resolve/main/"
    output_dir = Path("./QwQ-32B")
    output_dir.mkdir(exist_ok=True)
    
    num_shards = 14
    
    for i in range(1, num_shards + 1):
        # Note: The URL uses 'model-00001-of-00014.safetensors' format, not zero-padded total shards.
        filename = f"model-{i:05d}-of-00014.safetensors"
        download_url = f"{base_url}{filename}?download=true"
        output_path = output_dir / filename
        
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

if __name__ == "__main__":
    download_files()
