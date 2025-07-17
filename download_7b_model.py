import requests
from tqdm import tqdm
import os
from pathlib import Path

def download_7b_model():
    """
    Downloads the 4 shards of the QwQ-LCoT-7B-Instruct model from Hugging Face.
    """
    # --- Configuration ---
    model_repo = "prithivMLmods/QwQ-LCoT-7B-Instruct"
    num_shards = 4
    output_dir_name = "QwQ-LCoT-7B-Instruct"
    # ---------------------

    base_url = f"https://huggingface.co/{model_repo}/resolve/main/"
    output_dir = Path(f"./{output_dir_name}")
    output_dir.mkdir(exist_ok=True)
    
    print(f"--- Starting Download for {model_repo} ---")
    print(f"Files will be saved in: {output_dir.resolve()}")
    
    for i in range(1, num_shards + 1):
        filename = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
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
    
    print("\n--- Download Complete ---")

if __name__ == "__main__":
    # Ensure you have the necessary libraries: pip install requests tqdm
    download_7b_model()
