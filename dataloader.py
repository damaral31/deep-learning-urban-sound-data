import soundata
from soundata.core import Clip
import numpy
from pathlib import Path
import os
from typing import Callable
import time
from preprocessing import AudioPreprocessor
import config

class Dataloader():
    
    def __init__(self, dataset_path : str, preprocessing : AudioPreprocessor = None, verbose : bool = False, include_augmented : bool = False):
        self.dataset_path : Path = Path(dataset_path) / "urbansound8K"
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.include_augmented = include_augmented
        self.dataset = soundata.initialize("urbansound8k", data_home=self.dataset_path)
        self.all_clips = self.dataset.load_clips()
        self.n_original_clips = len(self.all_clips)
        self.clip_ids = list(self.all_clips.keys())
        
        if include_augmented:
            self.augmented_path = config.PROJECT_ROOT / "datasets" / "augmentation"
            self.n_augmented_clips = 0
            
            for root, dirs, files in os.walk(self.augmented_path):
                for f in files:
                    if f.endswith('.npy'):
                        self.n_augmented_clips += 1
        
        if self.verbose: print(f"Dataset loaded with {len(self)} clips\n")
    
    def no_preprocessing(self, clip : Clip): # recebe um objeto soudata.core.Clip
        return clip
    
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
    
    def __getitem__(self, i):
        
        clip_id = self.clip_ids[i]
        clip : Clip = self.all_clips[clip_id]
        
        if self.preprocessing is not None:
            treated_audio = self.preprocessing.process_clip(clip)
        else:  treated_audio = self.no_preprocessing(clip)
        
        if self.verbose:
            print(f"Item of index {i}")
            print(f"Clip ID: {clip.clip_id}")
            print(f"Fold: {clip.fold}")
            print(f"Class ID: {clip.class_id}")
            print(f"Class Label: {clip.class_label}")
            print(f"Salience: {clip.salience}")
            print("="*30)
        
        return treated_audio, clip.class_id # retorna um int q esta mapeado para um label
    

if __name__ == "__main__":
    dl = Dataloader(dataset_path=r"C:\Users\migue\Documents\MyCode\AC2\deep-learning-urban-sound-data\datasets",
                    verbose=True, include_augmented=True)
    l = len(dl)
    print(f"Length of dataloader: {l}\n")