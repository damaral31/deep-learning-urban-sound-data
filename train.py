"""
Script de Treino - Urban Sound 8K Deep Learning

Este script treina todos os modelos implementados:
- CNN (Convolutional Neural Network)
- RNN (Recurrent Neural Network)
- GRU (Gated Recurrent Unit) - RECOMENDADO
- BiRNN (Bidirectional RNN)
- LSTM (Long Short-Term Memory)
- LSTM+Attention (LSTM com mecanismo de atenção)

Uso:
    python train.py --model cnn          # Treinar CNN
    python train.py --model gru          # Treinar GRU
    python train.py --model all          # Treinar todos
    python train.py --epochs 50          # Customizar épocas
    python train.py --batch-size 64      # Customizar batch size
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Importar configuração
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROJECT_ROOT, MODELS_DIR, LOGS_DIR, DEVICE,
    DATASET, TRAINING, DATA_AUGMENTATION
)

# Importar modelos
from models.CNN import SoundCNN
from models.RNN import SoundRNN, SoundGRU, SoundBiRNN
from models.LSTM import SoundLSTM, SoundLSTMAttention


# ============================================================================
# DATASET SIMULADO
# ============================================================================

class UrbanSoundDataset(Dataset):
    """
    Dataset simulado Urban Sound 8K.
    
    Em produção, isso carregaria MEL espectrogramas reais do dataset.
    Atualmente simula dados realistas com características por classe.
    
    Atributos:
        num_samples: Número de amostras por classe
        num_classes: Número de categorias (10)
        height: Dimensão altura MEL spectrogram (40)
        width: Dimensão largura MEL spectrogram (174)
        augment: Se aplica data augmentation
    """
    
    # 10 categorias Urban Sound 8K
    CLASSES = [
        'Air Conditioner',      # 0 - ruído de ar condicionado
        'Car Horn',             # 1 - buzina de carro
        'Children Playing',     # 2 - crianças brincando
        'Dog Barking',          # 3 - cão a latir
        'Drilling',             # 4 - broca/furado
        'Engine Idling',        # 5 - motor ao ralenti
        'Gun Shot',             # 6 - disparo
        'Jackhammer',           # 7 - martelo pneumático
        'Siren',                # 8 - sirene
        'Street Music'          # 9 - música de rua
    ]
    
    def __init__(self, num_samples=100, augment=False):
        """
        Args:
            num_samples: Amostras por classe (default 100 = 1000 total)
            augment: Aplicar data augmentation (pitch shift, time stretch, etc)
        """
        self.num_samples = num_samples
        self.num_classes = len(self.CLASSES)
        self.height = DATASET['mel_bins']      # 40
        self.width = DATASET['mel_frames']     # 174
        self.augment = augment
        
        # Gerar dados
        self.data = []
        self.labels = []
        self._generate_data()
    
    def _generate_data(self):
        """Gera dados simulados realistas com características por classe."""
        
        for class_idx in range(self.num_classes):
            for _ in range(self.num_samples):
                # Spectrogram MEL (40 x 174)
                spec = self._generate_class_spectrogram(class_idx)
                self.data.append(spec)
                self.labels.append(class_idx)
    
    def _generate_class_spectrogram(self, class_idx):
        """
        Gera spectrogram realista com características da classe.
        
        Cada classe tem padrão único:
        - Air Conditioner: padrão uniforme (ruído rosa)
        - Car Horn: pico de frequência alta
        - Children Playing: padrão aleatório
        - Dog Barking: pulsos periódicos
        - Drilling: padrão periódico em frequências altas
        - Engine Idling: energia baixas frequências
        - Gun Shot: impulso explosivo
        - Jackhammer: padrão periódico rígido
        - Siren: varredura de frequência
        - Street Music: múltiplas componentes
        """
        
        spec = np.random.randn(self.height, self.width) * 0.1
        
        if class_idx == 0:  # Air Conditioner - ruído uniforme
            spec += np.random.randn(self.height, self.width) * 0.5
        
        elif class_idx == 1:  # Car Horn - frequência alta
            spec[30:40, :] += np.random.randn(10, self.width) * 0.8
        
        elif class_idx == 2:  # Children Playing - aleatório
            spec += np.random.randn(self.height, self.width) * 0.6
        
        elif class_idx == 3:  # Dog Barking - pulsos
            for x in range(10, self.width, 20):
                x_end = min(x + 5, self.width)
                width = x_end - x
                spec[:, x:x_end] += np.random.randn(self.height, width) * 0.7
        
        elif class_idx == 4:  # Drilling - periódico alto
            spec[25:40, :] += 0.5
            for x in range(0, self.width, 15):
                x_end = min(x + 3, self.width)
                width = x_end - x
                spec[25:40, x:x_end] += 0.5
        
        elif class_idx == 5:  # Engine Idling - baixas frequências
            spec[0:15, :] += 0.6
        
        elif class_idx == 6:  # Gun Shot - impulso
            mid = self.width // 2
            spec[10:35, mid-5:mid+5] += 1.0
        
        elif class_idx == 7:  # Jackhammer - periódico rígido
            for x in range(5, self.width, 10):
                x_end = min(x + 5, self.width)
                spec[20:40, x:x_end] += 0.8
        
        elif class_idx == 8:  # Siren - varredura frequência
            for x in range(self.width):
                freq = 10 + int(30 * x / self.width)
                freq = min(freq, self.height - 1)
                spec[freq, x] += 0.9
        
        elif class_idx == 9:  # Street Music - múltiplas
            spec[5:15, :] += 0.4    # Baixo
            spec[20:30, :] += 0.3   # Médio
            spec[35:40, :] += 0.2   # Alto
        
        # Normalizar
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        
        return spec.astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Retorna (spectrogram, label) em formato torch."""
        spec = torch.FloatTensor(self.data[idx]).unsqueeze(0)  # [1, 40, 174]
        label = torch.LongTensor([self.labels[idx]])[0]
        return spec, label


# ============================================================================
# TRAINER
# ============================================================================

class ModelTrainer:
    """
    Treinador de modelos com logging, checkpoints e visualizações.
    
    Funcionalidades:
    - Early stopping
    - Learning rate scheduling
    - Checkpoint dos melhores modelos
    - Logging em arquivo
    - Gráficos de treino
    """
    
    def __init__(self, model, model_name, device=DEVICE, 
                 lr=TRAINING['learning_rate'],
                 patience=TRAINING['early_stopping_patience']):
        """
        Args:
            model: Modelo PyTorch
            model_name: Nome para logging
            device: 'cuda' ou 'cpu'
            lr: Learning rate
            patience: Épocas sem melhorar antes de parar
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.patience = patience
        self.best_acc = 0
        self.patience_counter = 0
        
        # Otimizador e Loss
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        # Histórico
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Criar diretórios
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        self.model_path = os.path.join(MODELS_DIR, f'{model_name}.pt')
        self.log_path = os.path.join(LOGS_DIR, f'{model_name}.log')
    
    def _log(self, message):
        """Log em console e arquivo."""
        print(message)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    def train_epoch(self, train_loader):
        """Treina uma época."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Train [{self.model_name}]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Métricas
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader), correct / total
    
    def validate(self, val_loader):
        """Valida o modelo."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Valid [{self.model_name}]")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(val_loader), correct / total
    
    def fit(self, train_loader, val_loader, epochs):
        """Treina o modelo por N épocas."""
        
        self._log(f"\n{'='*80}")
        self._log(f"🚀 TREINO: {self.model_name}")
        self._log(f"{'='*80}")
        self._log(f"Modelo: {self.model_name}")
        self._log(f"Device: {self.device}")
        self._log(f"Épocas: {epochs}")
        self._log(f"Batch size: {train_loader.batch_size}")
        self._log(f"Train samples: {len(train_loader.dataset)}")
        self._log(f"Val samples: {len(val_loader.dataset)}")
        self._log(f"{'='*80}\n")
        
        for epoch in range(1, epochs + 1):
            # Treino
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validação
            val_loss, val_acc = self.validate(val_loader)
            
            # Log
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            message = (
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )
            self._log(message)
            
            # Learning rate scheduler
            self.scheduler.step(val_acc)
            
            # Early stopping com checkpoint
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.patience_counter = 0
                self._save_checkpoint()
                self._log(f"   ✅ Novo melhor modelo! (Acc: {val_acc:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self._log(f"\n⏹️  Early stopping! (patience: {self.patience})")
                    break
        
        self._log(f"\n✅ Treino concluído!")
        self._log(f"   Melhor acurácia: {self.best_acc:.4f}")
        self._log(f"   Modelo salvo em: {self.model_path}\n")
        
        # Gráficos
        self._plot_history()
    
    def _save_checkpoint(self):
        """Salva checkpoint do modelo."""
        torch.save({
            'model_state': self.model.state_dict(),
            'best_acc': self.best_acc,
            'history': self.history
        }, self.model_path)
    
    def _plot_history(self):
        """Plota gráficos de treino."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{self.model_name} - Loss')
        axes[0].legend()
        axes[0].grid()
        
        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'{self.model_name} - Accuracy')
        axes[1].legend()
        axes[1].grid()
        
        plot_path = os.path.join(LOGS_DIR, f'{self.model_name}_history.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        print(f"📊 Gráfico salvo em: {plot_path}")


# ============================================================================
# MAIN
# ============================================================================

def train_model(model_class, model_name, num_epochs, batch_size):
    """
    Treina um modelo completo.
    
    Args:
        model_class: Classe do modelo (SoundCNN, SoundRNN, etc)
        model_name: Nome para logging
        num_epochs: Número de épocas
        batch_size: Tamanho do batch
    """
    
    print(f"\n{'='*80}")
    print(f"📦 Preparando dados para {model_name}")
    print(f"{'='*80}\n")
    
    # Dataset
    dataset = UrbanSoundDataset(
        num_samples=100,  # 100 * 10 classes = 1000 amostras
        augment=DATA_AUGMENTATION['enabled']
    )
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows compatível
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✅ Dataset criado:")
    print(f"   Total: {len(dataset)} amostras")
    print(f"   Train: {len(train_dataset)} amostras")
    print(f"   Val: {len(val_dataset)} amostras")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {DEVICE}\n")
    
    # Modelo
    print(f"🧠 Inicializando {model_name}...")
    model = model_class()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parâmetros: {num_params:,}\n")
    
    # Treino
    trainer = ModelTrainer(model, model_name, device=DEVICE)
    trainer.fit(train_loader, val_loader, num_epochs)


def main():
    """Função principal."""
    
    parser = argparse.ArgumentParser(
        description="Treina modelos Deep Learning para Urban Sound 8K"
    )
    parser.add_argument(
        '--model', 
        default='all',
        choices=['cnn', 'rnn', 'gru', 'birnn', 'lstm', 'lstm_attn', 'all'],
        help='Qual modelo treinar (default: all)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=TRAINING['epochs'],
        help=f'Número de épocas (default: {TRAINING["epochs"]})'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=TRAINING['batch_size'],
        help=f'Batch size (default: {TRAINING["batch_size"]})'
    )
    
    args = parser.parse_args()
    
    # Modelos a treinar
    models = {
        'cnn': (SoundCNN, 'SoundCNN'),
        'rnn': (SoundRNN, 'SoundRNN'),
        'gru': (SoundGRU, 'SoundGRU'),
        'birnn': (SoundBiRNN, 'SoundBiRNN'),
        'lstm': (SoundLSTM, 'SoundLSTM'),
        'lstm_attn': (SoundLSTMAttention, 'SoundLSTMAttention'),
    }
    
    if args.model == 'all':
        to_train = list(models.values())
    else:
        to_train = [models[args.model]]
    
    print(f"\n{'='*80}")
    print(f"🎵 TREINO - URBAN SOUND 8K DEEP LEARNING")
    print(f"{'='*80}")
    print(f"Modelos a treinar: {len(to_train)}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {DEVICE}")
    print(f"{'='*80}\n")
    
    # Treinar cada modelo
    for model_class, model_name in to_train:
        train_model(model_class, model_name, args.epochs, args.batch_size)
        print()
    
    print(f"\n{'='*80}")
    print(f"✅ TODOS OS TREINOS CONCLUÍDOS!")
    print(f"{'='*80}")
    print(f"\n📁 Modelos salvos em: {MODELS_DIR}")
    print(f"📊 Logs salvos em: {LOGS_DIR}")
    print(f"\nPróximos passos:")
    print(f"  1. Verificar logs: ls -la {LOGS_DIR}")
    print(f"  2. Usar melhor modelo: python predict.py --model SoundGRU")
    print(f"\n")


if __name__ == "__main__":
    main()
