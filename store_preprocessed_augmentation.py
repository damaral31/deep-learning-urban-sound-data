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

def process_augmented_clip(args):
    """Process a single augmented clip and return metadata entry"""
    i, dl, CACHE_DIR, existing_filenames = args
    try:
        # Get processed data
        features, fold, label = dl[i]
        
        # Get metadata for this augmented clip
        aug_index = i - dl.n_original_clips
        aug_metadata = dl.augmented_metadata[aug_index]
        original_filename = aug_metadata['filename'] # e.g., "shift_135776-2-0-49.npy"
        
        # The filename already starts with the augmentation used (e.g. shift_, bandpass_)
        filename = original_filename
        
        # Create fold directory inside cache
        fold_dir = CACHE_DIR / f"fold{fold}"
        fold_dir.mkdir(exist_ok=True, parents=True)
        
        file_path = fold_dir / filename
        
        # Save to .npy
        # Saving as an object array containing [features, fold, label]
        data_to_save = np.array([features, fold, label], dtype=object)
        np.save(file_path, data_to_save)
        
        # Return metadata entry if not already present
        if filename not in existing_filenames:
            # Path should be relative to PROJECT_ROOT
            relative_path = file_path.relative_to(config.PROJECT_ROOT)
            
            metadata_entry = {
                "filename": filename,
                "fold": int(fold),
                "path": str(relative_path),
                "type": "augmented"
            }
            return metadata_entry
        return None
        
    except Exception as e:
        print(f"Error processing index {i}: {e}")
        return None

def main():
    # Define paths
    CACHE_DIR = config.PROJECT_ROOT / "datasets" / "augmentation_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    METADATA_FILE = CACHE_DIR / "augmentation_cached_files_index.json"

    # Load existing metadata if it exists
    metadata_list = []
    if METADATA_FILE.exists():
        print(f"Loading existing metadata from {METADATA_FILE}...")
        with open(METADATA_FILE, 'r') as f:
            try:
                metadata_list = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Metadata file is empty or corrupted. Starting fresh.")
                metadata_list = []
    else:
        print("No existing metadata found. Starting fresh.")

    # Initialize Preprocessor and Dataloader
    print("Initializing Preprocessor and Dataloader...")
    # Using default parameters as per previous context (sample_rate=22050, to_mono=True)
    preprocessor = AudioPreprocessor(sample_rate=22050, to_mono=True)
    
    # We want to process augmented files. Dataloader loads original + augmented if include_augmented=True.
    dataset_path = config.PROJECT_ROOT / "datasets"
    dl = Dataloader(dataset_path=str(dataset_path), preprocessing=preprocessor, verbose=False, include_augmented=True)

    print(f"Total clips in dataloader: {len(dl)}")
    print(f"Original clips: {dl.n_original_clips}")
    print(f"Augmented clips: {dl.n_augmented_clips}")
    
    start_index = dl.n_original_clips
    end_index = len(dl)
    
    print(f"Starting processing of {dl.n_augmented_clips} augmented clips...")
    
    # Create a set of existing filenames to avoid duplicates in metadata if re-running
    existing_filenames = {item['filename'] for item in metadata_list}
    
    # Use ThreadPoolExecutor for parallel processing
    max_workers = min(16, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_augmented_clip, (i, dl, CACHE_DIR, existing_filenames)) 
                   for i in range(start_index, end_index)]
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(futures), total=dl.n_augmented_clips):
            result = future.result()
            if result is not None:
                metadata_list.append(result)

    # Save updated metadata
    print(f"Saving updated metadata to {METADATA_FILE}...")
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata_list, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()