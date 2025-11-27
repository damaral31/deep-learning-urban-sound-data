import os
import numpy as np
import librosa
from scipy.signal import butter, lfilter
import csv
import pandas as pd


DATA_ROOT = "datasets"
AUGMENTATION_DIR = os.path.join(DATA_ROOT, "augmentation")
os.makedirs(AUGMENTATION_DIR, exist_ok=True)

# Arquivo CSV para guardar metadados
METADATA_CSV = os.path.join(AUGMENTATION_DIR, "augmented_metadata.csv")

# Carregar distribuição de classes
CLASS_DISTRIBUTION = pd.read_csv("class_distribution.csv")


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
#  CLASS BALANCING LOGIC
# ============================================================

def calculate_augmentation_needs():
    """
    Calcula quantas augmentações extras cada classe precisa para balancear.
    Retorna dict: {class_id: extras_por_audio}
    """
    max_count = CLASS_DISTRIBUTION['count'].max()
    target_count = max_count  # Alvo é a classe com mais exemplos
    
    augmentation_needs = {}
    
    for _, row in CLASS_DISTRIBUTION.iterrows():
        class_id = int(row['class_id'])
        current_count = int(row['count'])
        
        # Após augmentação básica (3x por áudio original): current_count * 4
        after_basic = current_count * 4
        
        # Quantos extras precisamos?
        target = target_count * 4
        needed = target - after_basic
        
        # Quantos extras por áudio original?
        extras_per_audio = needed / current_count if current_count > 0 else 0
        
        augmentation_needs[class_id] = max(0, int(np.ceil(extras_per_audio)))
    
    return augmentation_needs


# ============================================================
#  MAIN AUGMENTATION HANDLER
# ============================================================

def apply_augmentation(sound, augmentation_needs):
    """
    sound deve ser um objeto contendo:
    - sound.audio_path     → caminho do arquivo
    - sound.fold           → fold do dataset
    - sound.class_id       → label numérica
    - sound.slice_file_name → nome do arquivo
    
    augmentation_needs: dict com quantos extras por classe
    """

    y, sr = librosa.load(sound.audio_path, sr=None)

    # lista de augmentations básicas disponíveis
    augmentations = {
        "shift": time_shift(y, sr),
        "white_noise": add_white_noise(y),
        "bandpass": butter_bandpass_filter(y, sr)
    }

    # Criar diretório destino
    out_dir = os.path.join(AUGMENTATION_DIR, f"fold{sound.fold}")
    os.makedirs(out_dir, exist_ok=True)

    # Augmentação extra para balanceamento
    class_id = sound.class_id
    extras_needed = augmentation_needs.get(class_id, 0)

    # Gerar arquivos aumentados básicos
    for aug_name, aug_y in augmentations.items():
        # Mudar extensão de .wav para .npy
        base_name = os.path.splitext(sound.slice_file_name)[0]
        filename = f"{aug_name}_{base_name}.npy"
        out_path = os.path.join(out_dir, filename)
        
        # Guardar como objeto Python (tupla) usando allow_pickle
        np.save(out_path, np.array((aug_y, sr, class_id), dtype=object))
        print(f"[OK] Saved {out_path}")
    
    
    
    if extras_needed > 0:
        for i in range(extras_needed):
            # Usar diferentes thresholds de ruído branco
            noise_factor = 0.003 + (i * 0.0005)  # Varia o threshold
            aug_y = add_white_noise(y, noise_factor=noise_factor)
            
            base_name = os.path.splitext(sound.slice_file_name)[0]
            filename = f"extra_noise_{i}_{base_name}.npy"
            out_path = os.path.join(out_dir, filename)
            
            np.save(out_path, np.array((aug_y, sr, class_id), dtype=object))
            print(f"[OK] Saved extra {out_path}")


if __name__ == "__main__":
    from dataloader import Dataloader
    import config as cfg
    
    # Calcular necessidades de augmentação
    print("Calculando necessidades de balanceamento...")
    augmentation_needs = calculate_augmentation_needs()
    
    print("\nAugmentações extras por classe:")
    for class_id, extras in augmentation_needs.items():
        class_name = CLASS_DISTRIBUTION[CLASS_DISTRIBUTION['class_id'] == class_id]['class_name'].values[0]
        current_count = CLASS_DISTRIBUTION[CLASS_DISTRIBUTION['class_id'] == class_id]['count'].values[0]
        print(f"  Classe {class_id} ({class_name}): {current_count} originais → +{extras} extras por áudio")
    
    print("\n" + "="*60)
    print("Iniciando augmentação...")
    print("="*60 + "\n")
    
    dl = Dataloader(r"C:\\Users\\diogo\\OneDrive\\Documents", verbose=False)
    
    l = len(dl)
    for i in range(l):
        clip_id = dl.clip_ids[i]
        sound = dl.all_clips[clip_id]
        
        print(f"\n[{i+1}/{l}] Processando {sound.slice_file_name} (classe {sound.class_id})...")
        apply_augmentation(sound, augmentation_needs)
    
    print("\n" + "="*60)
    print("Augmentação completa!")
    print("="*60)
    
    # Calcular e guardar nova distribuição
    print("\nCalculando nova distribuição de classes...")
    new_distribution = []
    
    for _, row in CLASS_DISTRIBUTION.iterrows():
        class_id = int(row['class_id'])
        class_name = row['class_name']
        original_count = int(row['count'])
        
        # Augmentação básica: 3 augmentações + original = 4x
        basic_augmented = original_count * 4
        
        # Augmentação extra
        extras = augmentation_needs[class_id] * original_count
        
        # Total
        new_count = basic_augmented + extras
        
        new_distribution.append({
            'class_id': class_id,
            'class_name': class_name,
            'count': new_count,
            'proportion': 0  # Será calculado depois
        })
    
    # Calcular proporções
    total_samples = sum(d['count'] for d in new_distribution)
    for d in new_distribution:
        d['proportion'] = round(d['count'] / total_samples, 4)
    
    # Guardar CSV
    new_df = pd.DataFrame(new_distribution)
    output_csv = "augmented_class_distribution.csv"
    new_df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Nova distribuição guardada em '{output_csv}'")
    print("\nNova distribuição:")
    print(new_df.to_string(index=False))
        


