"""
Configurações Globais - Urban Sound 8K Deep Learning Project

Este arquivo centraliza todas as configurações do projeto:
- Caminhos de dados
- Parâmetros de dataset
- Configurações de modelo
- Hiperparâmetros de treino
- Configurações de augmentation
"""

import os
from pathlib import Path
import torch

# ============================================================================
# CAMINHOS
# ============================================================================

# Raiz do projeto
PROJECT_ROOT = Path(__file__).parent.absolute()

# Diretórios
DATA_DIR = PROJECT_ROOT / "datasets" / "UrbanSound8K"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"

# Caminho alternativo (se dataset estiver em outro local)
DATASET_PATH = 'C:\\Users\\diogo\\OneDrive\\Documents\\UrbanSound8K\\UrbanSound8K\\audio'

# Criar diretórios se não existirem
for directory in [MODELS_DIR, LOGS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DEVICE
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# DATASET - Urban Sound 8K
# ============================================================================

DATASET = {
    'name': 'UrbanSound8K',
    'num_classes': 10,
    'num_samples': 8732,
    'sample_rate': 48000,
    'duration': 4.0,
    'mel_bins': 40,
    'mel_frames': 174,  # Número de frames temporais no spectrogram
    'n_fft': 2048,
    'hop_length': 512,
    'classes': {
        0: 'air_conditioner',
        1: 'car_horn',
        2: 'children_playing',
        3: 'dog_barking',
        4: 'drilling',
        5: 'engine_idling',
        6: 'gun_shot',
        7: 'jackhammer',
        8: 'siren',
        9: 'street_music',
    }
}

# ============================================================================
# TREINO
# ============================================================================

TRAINING = {
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'early_stopping_patience': 15,
}

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

DATA_AUGMENTATION = {
    'enabled': True,
    'time_stretch': True,
    'pitch_shift': True,
    'dynamic_range_compression': True,
    'background_noise': True,
}

# ============================================================================
# SEED
# ============================================================================

SEED = 42