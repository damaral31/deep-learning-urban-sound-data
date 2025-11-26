import os
import glob

import os
import glob
import numpy as np
import torch
import librosa

class UrbanSoundNPYLoader(torch.utils.data.Dataset):
    def __init__(self, base_path, folders, pad_length=88200, n_mels=64, shuffle=True):
        self.files = []
        for folder in folders:
            folder_path = os.path.join(base_path, folder)
            self.files.extend(glob.glob(os.path.join(folder_path, "*.npy")))
        self.pad_length = pad_length
        self.n_mels = n_mels
        if shuffle:
            np.random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx], allow_pickle=True)
        audio, class_id, sr = arr[0], arr[1], arr[2]
        # Zero-pad or crop audio to pad_length
        if len(audio) < self.pad_length:
            pad_width = self.pad_length - len(audio)
            audio = np.pad(audio, (0, pad_width), mode='constant')
        elif len(audio) > self.pad_length:
            audio = audio[:self.pad_length]
        # Normaliza áudio para [-1, 1] para evitar overflow
        audio = np.asarray(audio, dtype=np.float32)
        max_abs = np.max(np.abs(audio))
        if max_abs > 0:
            audio = audio / max_abs
        # Substitui NaN e Inf por zero no áudio
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # Normaliza para [0, 1] com proteção contra divisão por zero
        min_val = mel_spec_db.min()
        max_val = mel_spec_db.max()
        denom = max(max_val - min_val, 1e-6)
        mel_spec_db = (mel_spec_db - min_val) / denom
        # Substitui NaN e Inf por zero
        mel_spec_db = np.nan_to_num(mel_spec_db, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)  # [1, n_mels, time]
        y = torch.tensor(int(class_id), dtype=torch.long)
        return x, y