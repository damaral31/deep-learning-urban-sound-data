import os
import librosa
import numpy as np
from soundata.core import Clip
import config
import librosa.display
import matplotlib.pyplot as plt

class AudioPreprocessor:
    def __init__(self, sample_rate=22050, to_mono=True):
        self.sample_rate = sample_rate
        self.to_mono = to_mono
        
        self.save_path = config.PROJECT_ROOT
        self.target_length : int = self.sample_rate * config.NORMALIZED_AUDIO_LENGTH
        
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
        original = self._extract_log_mel(audio)
        delta = self._extract_delta(original)
        delta_delta = self._extract_delta_delta(original)
        harmonic = self._extract_hpss_harmonic(audio)
        percussive = self._extract_hpss_percussive(audio)
        
        return np.stack([original, delta, delta_delta, harmonic, percussive])

    def _extract_log_mel(self, audio : np.ndarray):
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=64)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        return log_mel_spectrogram
    
    def _extract_delta(self, data : np.ndarray):
        return librosa.feature.delta(data)
    
    def _extract_delta_delta(self, data : np.ndarray):
        return librosa.feature.delta(data, order=2)
    
    def _extract_hpss_harmonic(self, audio : np.ndarray):
        y_harmonic = librosa.effects.hpss(audio)[0]
        return self._extract_log_mel(y_harmonic)
    
    def _extract_hpss_percussive(self, audio : np.ndarray):
        y_percussive = librosa.effects.hpss(audio)[1]
        return self._extract_log_mel(y_percussive)
    
    def process_clip(self, audio : np.ndarray, original_sr : int):
        audio = self.monorize_audio(audio)
        audio = self.resample_audio(audio, original_sr)
        audio = self.zero_pad_audio(audio)
        features = self.extract_features(audio)
        return features

    
    def plot_all_channels(self, features : np.ndarray, fold : int = None, label : str = None):
        
        title = "Audio Features"
        
        if fold is not None and label is not None:
            title += f" (Label: {label}, Fold: {fold})"
        elif fold is not None:
            title += f" (Fold: {fold})"
        elif label is not None:
            title += f" (Label: {label})"
        
        titles = ['Log Mel Spectrogram', 'Delta', 'Delta-Delta', 'Harmonic', 'Percussive']
        fig, axes = plt.subplots(5, 1, figsize=(10, 15))
        fig.suptitle(title)
        
        for i, ax in enumerate(axes):
            img = librosa.display.specshow(features[i], x_axis='time', y_axis='mel', sr=self.sample_rate, ax=ax)
            ax.set_title(titles[i])
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    import soundata

    # Example usage
    preprocessor = AudioPreprocessor()
    
    # Initialize dataset directly without Dataloader
    dataset_path = r"E:\deep-learning-urban-sound-data\datasets\urbansound8k"
    dataset = soundata.initialize("urbansound8k", data_home=dataset_path)
    
    # Get all clips
    all_clips = dataset.load_clips()
    
    # Find a gun_shot clip (class_id 6 is gun_shot)
    gun_shot_clips = [clip_id for clip_id, clip in all_clips.items() if clip.class_label == 'gun_shot']
    
    if gun_shot_clips:
        # Take the first one found
        clip_id = gun_shot_clips[0]
        clip = dataset.clip(clip_id)
        
        print(f"Processing clip: {clip_id} (Label: {clip.class_label})")
        
        audio, sr = clip.audio
        features = preprocessor.process_clip(audio, sr)
        
        print(f"Features shape: {features.shape}")
        
        preprocessor.plot_all_channels(features, 1, "gun_shot")
    else:
        print("No gun_shot clips found.")