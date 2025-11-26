import os
import librosa
import numpy as np
from soundata.core import Clip
import config

class AudioPreprocessor:
    def __init__(self, sample_rate=22050, to_mono=True, extraction_method='log-mel'):
        self.sample_rate = sample_rate
        self.to_mono = to_mono
        self.extraction_method = extraction_method
        
        self.save_path = config.PROJECT_ROOT
        self.target_length : int = self.sample_rate * config.NORMALIZED_AUDIO_LENGTH
        
        if self.extraction_method not in ["log-mel", "mfcc"]:
            raise ValueError("Invalid extraction method. Choose 'log-mel' or 'mfcc'.")
    
    def monorize_audio(self, audio : np.ndarray):
        if self.to_mono:
            audio = librosa.to_mono(audio)
        return audio

    def resample_audio(self, audio : np.ndarray, original_sr : int):
        if original_sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.sample_rate)
        return audio

    def zero_pad_audio(self, audio : np.ndarray):
        target_length = int(self.target_length)
        current_length = audio.shape[-1]
        
        if current_length > target_length:
            audio = audio[..., :target_length]
        elif current_length < target_length:
            padding = target_length - current_length
            audio = np.pad(audio, (0, padding), mode='constant')
            
        return audio
    
    def extract_features(self, audio : np.ndarray):
        if self.extraction_method == "log-mel":
            return self._extract_log_mel(audio)
        
        elif self.extraction_method == "mfcc":
            return self._extract_mfcc(audio)
    
    def _extract_log_mel(self, audio : np.ndarray):
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=64)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        return log_mel_spectrogram

    def _extract_mfcc(self, audio : np.ndarray):
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        return mfcc
    
    def process_clip(self, clip: Clip):
        audio, original_sr = clip.audio
        audio = self.monorize_audio(audio)
        audio = self.resample_audio(audio, original_sr)
        audio = self.zero_pad_audio(audio)
        features = self.extract_features(audio)
        return features


if __name__ == "__main__":
    # Exemplo de uso
    from soundata import initialize

    dataset_path = r"E:\deep-learning-urban-sound-data\urbansound8k"
    dataset = initialize("urbansound8k", data_home=dataset_path)

    # Teste com o primeiro clipe
    # Obter o primeiro clipe com a class Dataloader
    from dataloader import Dataloader
    dl = Dataloader(dataset_path)
    
    # Obter o primeiro clipe da label gunshot (class_id = 6)
    gunshot_clips = [clip for clip in dl.all_clips.values() if clip.class_id == 6]
    clip = gunshot_clips[0]
    
    preprocessor = AudioPreprocessor(sample_rate=22050, to_mono=True, extraction_method='log-mel')
    features = preprocessor.process_clip(clip)
    print(f"Extracted features shape: {features.shape}")
    
    # Mostrar o audio original e o processado
    import matplotlib.pyplot as plt
    
    # Obter o label do clipe
    label = clip.tags.labels[0] if clip.tags.labels else "Unknown"

    # Processar o áudio para obter a forma de onda processada (antes da extração de features)
    audio, original_sr = clip.audio
    audio_mono = preprocessor.monorize_audio(audio)
    audio_resampled = preprocessor.resample_audio(audio_mono, original_sr)
    audio_processed = preprocessor.zero_pad_audio(audio_resampled)

    plt.figure(figsize=(15, 8))
    plt.suptitle(f"Label: {label}", fontsize=16)

    # 1. Audio Original (Waveform)
    plt.subplot(2, 2, 1)
    plt.title("Original Audio (Waveform)")
    plt.plot(clip.audio[0]) # Plotando apenas o primeiro canal se for estéreo
    
    # 2. Audio Processado (Waveform)
    plt.subplot(2, 2, 2)
    plt.title("Processed Audio (Waveform)")
    plt.plot(audio_processed)

    # 3. Features (Log-Mel Spectrogram)
    plt.subplot(2, 1, 2)
    plt.title(f"Processed Features ({preprocessor.extraction_method})")
    plt.imshow(features, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.show()