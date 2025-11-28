import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import config
from dataloader import Dataloader
from preprocessing import AudioPreprocessor

def process_clip(args):
    """Process a single clip and return metadata entry"""
    i, dl, CACHE_DIR = args
    try:
        # Get processed data
        features, fold, label = dl[i]
        
        # Determine filename
        clip_id = dl.clip_ids[i]
        filename = f"original_{clip_id}.npy"
        
        # Create fold directory inside cache
        fold_dir = CACHE_DIR / f"fold{fold}"
        fold_dir.mkdir(exist_ok=True, parents=True)
        
        file_path = fold_dir / filename
        
        # Save to .npy
        # Saving as an object array containing [features, fold, label]
        data_to_save = np.array([features, fold, label], dtype=object)
        np.save(file_path, data_to_save)
        
        # Return metadata entry
        # Path should be relative to PROJECT_ROOT to be consistent with augmented_files_index.json
        relative_path = file_path.relative_to(config.PROJECT_ROOT)
        
        metadata_entry = {
            "filename": filename,
            "fold": int(fold),
            "path": str(relative_path),
            "type": "original"
        }
        return metadata_entry
        
    except Exception as e:
        print(f"Error processing index {i}: {e}")
        return None

def main():
    # Define paths
    CACHE_DIR = config.PROJECT_ROOT / "datasets" / "cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    METADATA_FILE = CACHE_DIR / "cached_files_index.json"

    # Initialize Preprocessor and Dataloader
    print("Initializing Preprocessor and Dataloader...")
    # Using default parameters as per previous context (sample_rate=22050, to_mono=True)
    preprocessor = AudioPreprocessor(sample_rate=22050, to_mono=True)
    
    # We want to cache only original files
    dataset_path = config.PROJECT_ROOT / "datasets"
    dl = Dataloader(dataset_path=str(dataset_path), preprocessing=preprocessor, verbose=False, include_augmented=False)

    metadata_list = []
    
    print(f"Starting processing of {len(dl)} clips...")
    
    # Use ThreadPoolExecutor for parallel processing
    max_workers = min(16, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_clip, (i, dl, CACHE_DIR)) for i in range(len(dl))]
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(futures), total=len(dl)):
            result = future.result()
            if result is not None:
                metadata_list.append(result)

    # Save metadata
    print(f"Saving metadata to {METADATA_FILE}...")
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata_list, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()