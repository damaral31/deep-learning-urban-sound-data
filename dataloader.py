import soundata
from soundata.core import Clip
import numpy
import os
from typing import Callable
import time
from preprocessing import AudioPreprocessor

class Dataloader():
    
    def __init__(self, dataset_path : str, preprocessing : AudioPreprocessor = None, verbose : bool = False):
        self.dataset_path = dataset_path
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.dataset = soundata.initialize("urbansound8k", data_home=dataset_path)
        self.all_clips = self.dataset.load_clips()
        self.clip_ids = list(self.all_clips.keys())
            
        if self.verbose: print(f"Dataset loaded with {len(self)} clips\n")
    
    def no_preprocessing(self, clip : Clip): # recebe um objeto soudata.core.Clip
        return clip
    
    def __len__(self):
        return len(self.all_clips)

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
    start = time.perf_counter()
    dl = Dataloader(r"C:\Users\migue\Documents\MyCode\AC2\urbansound8k", verbose=True)


    for i in range(len(dl)):
        _ = dl[i][0].audio # quando o preprocessing tiver feito tira se o .audio

    elapsed = time.perf_counter() - start
    print(f"Total elapsed time: {elapsed:.3f} seconds")