import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter
import csv


DATA_ROOT = "datasets"
AUGMENTATION_DIR = os.path.join(DATA_ROOT, "augmentation")
os.makedirs(AUGMENTATION_DIR, exist_ok=True)

# Arquivo CSV para guardar metadados
METADATA_CSV = os.path.join(AUGMENTATION_DIR, "augmented_metadata.csv")


# ============================================================
#  AUGMENTATION FUNCTIONS
# ============================================================

def time_shift(y, sr, shift_max=0.5):
    """Desloca o áudio no tempo em até shift_max segundos."""
    shift = int(sr * shift_max)
    shift = np.random.randint(-shift, shift)
    return np.roll(y, shift)


def add_white_noise(y, noise_factor=0.005):
    """Adiciona ruído branco."""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


def butter_bandpass_filter(y, sr, lowcut=300, highcut=3000, order=5):
    """Aplica filtro passa-banda."""
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, y)


# ============================================================
#  MAIN AUGMENTATION HANDLER
# ============================================================

def apply_augmentation(sound):
    """
    sound deve ser um objeto contendo:
    - sound.audio_path     → caminho do arquivo
    - sound.fold           → fold do dataset
    - sound.class_id       → label numérica
    - sound.slice_file_name → nome do arquivo
    """

    y, sr = librosa.load(sound.audio_path, sr=None)

    # lista de augmentations disponíveis
    augmentations = {
        "shift": time_shift(y, sr),
        "white_noise": add_white_noise(y),
        "bandpass": butter_bandpass_filter(y, sr)
    }

    # Criar diretório destino
    out_dir = os.path.join(AUGMENTATION_DIR, f"fold{sound.fold}", str(sound.class_id))
    os.makedirs(out_dir, exist_ok=True)

    # Lista para guardar metadados
    metadata_rows = []

    # Gerar arquivos aumentados
    for aug_name, aug_y in augmentations.items():
        filename = f"{aug_name}_{sound.slice_file_name}"
        out_path = os.path.join(out_dir, filename)

        sf.write(out_path, aug_y, sr)
        print(f"[OK] Saved {out_path}")
        
        # Guardar metadados
        metadata_rows.append({
            'filename': filename,
            'fold': sound.fold,
            'class_id': sound.class_id,
            'augmentation': aug_name,
            'original_file': sound.slice_file_name
        })
    
    return metadata_rows


if __name__ == "__main__":
    from dataloader import Dataloader
    import config as cfg
    dl = Dataloader("C:\\Users\\diogo\\OneDrive\\Documents", verbose=False)

    # Criar CSV com cabeçalho
    with open(METADATA_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'fold', 'class_id', 'augmentation', 'original_file'])
        writer.writeheader()

    l = len(dl)
    for i in range(l):
        clip, _ = dl[i]
        print(f"\n[{i+1}/{l}] Processing {clip.slice_file_name}")
        
        metadata = apply_augmentation(clip)
        
        # Adicionar ao CSV
        with open(METADATA_CSV, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'fold', 'class_id', 'augmentation', 'original_file'])
            writer.writerows(metadata)

        break
    
    print(f"\n[DONE] Metadata saved to {METADATA_CSV}")


