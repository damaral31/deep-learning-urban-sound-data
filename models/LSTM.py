"""
Módulo SoundLSTM - Rede Neural Recorrente (LSTM) para classificação de áudio urbano

Este módulo implementa um modelo LSTM especializado em capturar dependências temporais
em espectrogramas de sons urbanos (tráfego, sirenes, buzinas, etc).

Diferença CNN vs LSTM:
- CNN: Excelente para detectar padrões ESPACIAIS (bordas, formas)
- LSTM: Excelente para capturar DEPENDÊNCIAS TEMPORAIS (sequências, progressão)

Para áudio urbano, LSTM é melhor porque:
- Sons urbanos têm progressão temporal importante (início, meio, fim de um som)
- Uma sirene evolui ao longo do tempo
- O contexto passado ajuda a prever o próximo frame do áudio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoundLSTM(nn.Module):
    """
    Rede Neural Recorrente (LSTM) para classificação de sons urbanos.
    
    Arquitetura:
    1. Camada convolucional inicial: extrai features 2D do espectrograma
    2. Reshape em sequência temporal: cada coluna do espectrograma = um timestep
    3. LSTM bidirecional: processa sequência em ambas direções (forward + backward)
    4. Camadas fully-connected: classificação final
    
    Por que LSTM e não RNN simples?
    - RNN simples: sofre de "vanishing gradient" (esquece informações antigas)
    - LSTM: tem "gates" que controlam o que lembrar e o que esquecer
    - Perfeito para sequências longas como espectrogramas de áudio
    """
    
    def __init__(self, num_classes=10, input_height=40, input_width=174,
                 hidden_size=128, num_layers=2, dropout_rate=0.5, in_channels=1):
        """
        Inicializa o modelo LSTM.
        
        Args:
            num_classes (int): Número de categorias de áudio (padrão: 10)
            input_height (int): Altura do espectrograma (padrão: 40 - mel bins)
            input_width (int): Largura do espectrograma (padrão: 174 - frames)
            hidden_size (int): Dimensão do estado oculto LSTM (padrão: 128)
            num_layers (int): Número de camadas LSTM empilhadas (padrão: 2)
            dropout_rate (float): Taxa de dropout para regularização (padrão: 0.5)
        """
        super(SoundLSTM, self).__init__()
        
        # ========== CAMADA CONVOLUCIONAL INICIAL ==========
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.pool_conv = nn.MaxPool2d(kernel_size=(2, 1))
        
        # ========== CAMADA LSTM BIDIRECIONAL ==========
        # Cálculo do tamanho de entrada LSTM:
        # Após convução + pooling: [batch, 32 canais, altura//2, largura]
        # Cada timestep = 32 * (altura // 2) features
        lstm_input_size = 32 * (input_height // 2)
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # ========== CAMADAS FULLY-CONNECTED ==========
        # hidden_size * 2 porque LSTM é bidirecional
        # Forward + Backward = hidden_size + hidden_size
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # ========== REGULARIZAÇÃO ==========
        self.dropout = nn.Dropout(dropout_rate)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

    def forward(self, x):
        """
        Define o fluxo de dados através da rede (forward pass).
        
        Args:
            x (torch.Tensor): Input espectrograma com shape [batch_size, 1, altura, largura]
                            Exemplo: [8, 1, 40, 174]
                            
        Returns:
            torch.Tensor: Logits de saída com shape [batch_size, num_classes]
                         Exemplo: [8, 10]
        """
        batch_size = x.size(0)
        
        # ========== PASSO 1: CONVOLUÇÃO + BATCH NORM + ReLU ==========
        x = self.conv1(x)       # [batch, 32, 40, 174]
        x = self.bn_conv1(x)
        x = F.relu(x)
        
        # ========== PASSO 2: MAX POOLING (ALTURA) ==========
        x = self.pool_conv(x)   # [batch, 32, 20, 174]
        
        # ========== PASSO 3: RESHAPE PARA SEQUÊNCIA TEMPORAL ==========
        # Converte de [batch, 32, 20, 174] para [batch, 174, 640]
        # onde:
        # - 174 = timesteps (frames temporais)
        # - 640 = 32 canais × 20 altura (features por timestep)
        
        _, channels, height, time = x.shape
        
        # Permute: [batch, 32, 20, 174] → [batch, 174, 32, 20]
        x = x.permute(0, 3, 1, 2)
        
        # View: [batch, 174, 32, 20] → [batch, 174, 640]
        x = x.contiguous().view(batch_size, time, channels * height)
        
        # ========== PASSO 4: LSTM BIDIRECIONAL ==========
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: [batch, 174, 256]
        # h_n shape: [num_layers * 2, batch, 128]
        # c_n shape: [num_layers * 2, batch, 128]
        
        # ========== PASSO 5: EXTRAÇÃO DO CONTEXTO FINAL ==========
        # Usar o último output LSTM que contém contexto de toda sequência
        x = lstm_out[:, -1, :]  # [batch, 256]
        
        # ========== PASSO 6: PRIMEIRA CAMADA FULLY-CONNECTED ==========
        x = self.fc1(x)         # [batch, 128]
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # ========== PASSO 7: SEGUNDA CAMADA FULLY-CONNECTED (SAÍDA) ==========
        x = self.fc2(x)         # [batch, 10]
        
        return x