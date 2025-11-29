import soundata
from soundata.core import Clip
import numpy as np
from pathlib import Path
import os
from typing import Callable
import time
from preprocessing import AudioPreprocessor
import config
import json
import matplotlib.pyplot as plt

class Dataloader():
    
    def __init__(self, dataset_path : str, preprocessing : AudioPreprocessor = None,
                 include_augmented : bool = False, use_cache : bool = False,
                 folds : list[str] = None, verbose : bool = False):
        
        self.n_augmented_clips = 0
        self.dataset_path : Path = Path(dataset_path) / "urbansound8k"
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.include_augmented = include_augmented
        self.use_cache = use_cache
        self.folds = folds if folds is not None else [f"fold{i}" for i in range(1, 11)]
        # Garante que os folds estão no formato 'fold1', 'fold2', ...
        self.folds = [f if f.startswith('fold') else f"fold{f}" for f in self.folds]

        self.dataset = soundata.initialize("urbansound8k", data_home=self.dataset_path)
        all_clips = self.dataset.load_clips()
        # Filtra os clips para os folds desejados
        self.all_clips = {k: v for k, v in all_clips.items() if f"fold{v.fold}" in self.folds or v.fold in self.folds or str(v.fold) in self.folds or f"fold{str(v.fold)}" in self.folds}
        self.clip_ids = list(self.all_clips.keys())
        self.n_original_clips = len(self.all_clips)

        if self.use_cache:
            self.original_cache_path = config.PROJECT_ROOT / "datasets" / "cache"
            cache_metadata_path = self.original_cache_path / "cached_files_index.json"
            self.augmentation_cache_path = config.PROJECT_ROOT / "datasets" / "augmentation_cache"
            augmentation_cache_metadata_path = self.augmentation_cache_path / "augmentation_cached_files_index.json"
            
            self.cache_metadata = json.load(open(cache_metadata_path))
            self.cache_metadata = [m for m in self.cache_metadata if f"fold{m['fold']}" in self.folds]
            self.n_original_clips = len(self.cache_metadata)
            
            self.augmentation_cache  = json.load(open(augmentation_cache_metadata_path))
            self.augmentation_cache = [m for m in self.augmentation_cache if f"fold{m['fold']}" in self.folds]
            self.n_augmented_clips = len(self.augmentation_cache)
        else:
            if self.include_augmented:
                self.n_augmentation_path = config.PROJECT_ROOT / "datasets" / "augmentation"
                augmentation_metadata_path = self.n_augmentation_path / "augmented_files_index.json"
                self.augmentation_metadata = json.load(open(augmentation_metadata_path))
                self.augmentation_metadata = [m for m in self.augmentation_metadata if f"fold{m['fold']}" in self.folds]
                self.n_augmented_clips = len(self.augmentation_metadata)
        if self.verbose: print(f"Dataset loaded with {len(self)} clips (folds: {self.folds})\n")
    
    def no_preprocessing(self, audio : np.ndarray, sample_rate : int):
        return audio
    
    def __len__(self):
        if self.include_augmented:
            return self.n_original_clips + self.n_augmented_clips
        else:
            return self.n_original_clips

    def get_label_mapping(self):
        class_mapping = {
            0: 'air_conditioner',
            1: 'car_horn',
            2: 'children_playing',
            3: 'dog_bark',
            4: 'drilling',
            5: 'engine_idling',
            6: 'gun_shot',
            7: 'jackhammer',
            8: 'siren',
            9: 'street_music'
        }
        return class_mapping
    
    def plot_waveform(self, audio : np.ndarray):
        
        plt.figure(figsize=(10, 4))
        plt.plot(audio)
        plt.title("Audio Waveform")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.show()
    
    def _get_item_from_cache(self, i : int) -> tuple[np.ndarray, int]:
        
        if i < self.n_original_clips:
            metadata = self.cache_metadata[i]
        elif self.include_augmented and i >= self.n_original_clips:
            metadata = self.augmentation_cache[i - self.n_original_clips]

        file_path = config.PROJECT_ROOT / metadata['path']
        data = np.load(file_path, allow_pickle=True)
        
        features = data[0]
        fold = data[1]
        label = data[2]
        
        if self.verbose :
            if i < self.n_original_clips:
                print(f"Item of index {i} (from original cache)")
            else:
                print(f"Item of index {i} (from augmentation cache)")
            print(f"Loaded from: {file_path}")
            print(f"Fold: {fold}")
            print(f"Label: {label}")
            print("="*30)
        
        return features, fold, label
    
    def __getitem__(self, i : int) -> tuple[np.ndarray, int]:
        
        if self.use_cache:
            return self._get_item_from_cache(i)
        else: pass
        
        if i in range(0, self.n_original_clips):
            clip_id = self.clip_ids[i]
            clip : Clip = self.all_clips[clip_id]
            audio, sample_rate = clip.audio
            label = clip.class_id
            fold = clip.fold
            
            if self.verbose :
                print(f"Item of index {i} (original)")
                print(f"Clip ID: {clip.clip_id}")
                print(f"Fold: {clip.fold}")
                print(f"Class ID: {clip.class_id}")
                print(f"Class Label: {clip.class_label}")
                print(f"Salience: {clip.salience}")
                print("="*30)

        elif self.include_augmented and i >= self.n_original_clips:
            aug_index = i - self.n_original_clips
            meta = self.augmentation_metadata[aug_index]
            
            file_path = config.PROJECT_ROOT / meta['path']
            loaded_data = np.load(file_path, allow_pickle=True)
            
            # Check if it's the object array structure [audio, sr, label]
            if loaded_data.dtype == 'O' and loaded_data.shape == (3,):
                audio = loaded_data[0]
                sample_rate = int(loaded_data[1])
                label = int(loaded_data[2])
            else:
                audio = loaded_data
                sample_rate = 22050 # Assumption
                
                filename = meta['filename']
                # Extract label from filename (format: prefix_fsID-classID-occurrenceID-sliceID.npy)
                try:
                    label = int(filename.replace('.npy', '').split('-')[1])
                except IndexError:
                    print(f"Error parsing label from {filename}")
                    label = -1
            
            filename = meta['filename']
            fold = meta['fold']
            
            if self.verbose:
                print(f"Item of index {i} (augmented)")
                print(f"Filename: {filename}")
                print(f"Fold: {fold}")
                print(f"Label: {label}")
                print("="*30)
        
        if self.preprocessing is not None:
            treated_audio = self.preprocessing.process_clip(audio, sample_rate)
        else:  treated_audio = self.no_preprocessing(audio, sample_rate)
        
        return treated_audio, fold, label # retorna um int q esta mapeado para um label


if __name__ == "__main__":
    dl = Dataloader(dataset_path=r"C:\deep-learning-urban-sound-data\datasets",
                    verbose=True, include_augmented=True, use_cache=True,
                    preprocessing=AudioPreprocessor(), folds=['fold1'])
    
    l = len(dl)
    print(f"Length of dataloader:\n"
          f"Original clips: {dl.n_original_clips}\n"
          f"Augmented clips: {dl.n_augmented_clips}\n")
    
    #for i in range(l):
    #    aug_item, fold, label = dl[i]
    
    """
    aug_item shape:
    0: original log-mel
    1: delta
    2: delta-delta
    3: harmonic
    4: percussive
    """
    
    #str_label = dl.get_label_mapping()[label]
    #dl.preprocessing.plot_all_channels(aug_item, fold, str_label)