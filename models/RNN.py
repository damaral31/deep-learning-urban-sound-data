"""
Módulo SoundRNN - Rede Neural Recorrente para classificação de áudio urbano

Este módulo implementa modelos RNN (Recurrent Neural Network) para capturar
dependências temporais em espectrogramas de sons urbanos.

Tipos de RNN implementados:
1. SoundRNN: RNN simples (vanilla RNN)
2. SoundGRU: RNN com Gated Recurrent Units (mais eficiente que LSTM)
3. SoundBiRNN: RNN Bidirecional (combina contexto anterior e posterior)

Comparação: CNN vs RNN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                CNN         RNN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Força       Padrões      Sequências
            espaciais    temporais

Fraqueza    Sequências   Padrões
            longas       espaciais

Melhor      Imagens      Áudio, Texto,
para        básicas      Vídeo

Problema    Vanishing    Vanishing
            -            Gradient

Solução     N/A          LSTM/GRU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Por que RNN para áudio urbano?
- Sons urbanos têm estrutura temporal importante
- Uma sirene evolui: início → meio → fim
- O contexto passado ajuda a prever o futuro
- Perfil acústico varia ao longo do tempo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoundRNN(nn.Module):
    """
    Rede Neural Recorrente (RNN) Vanilla para classificação de sons urbanos.
    
    Uma RNN vanilla é a forma mais simples de rede recorrente:
    - Lê a sequência de entrada frame por frame
    - Mantém um estado oculto que atualiza a cada timestep
    - O estado oculto contém informação do passado
    
    Vantagens:
    - Simples e rápida
    - Fácil de entender e debugar
    
    Desvantagens:
    - Sofre de "vanishing gradient" em sequências longas
    - Esquece informações antigas rapidamente
    
    Fórmula RNN:
    h_t = tanh(W_ih * x_t + W_hh * h_(t-1) + b)
    
    Onde:
    - x_t: entrada no timestep t
    - h_t: estado oculto no timestep t
    - W_ih, W_hh: pesos (input-to-hidden, hidden-to-hidden)
    - tanh: ativação não-linear
    
    Fluxo:
    Input [batch, 1, 40, 174]
         ↓
    Conv [batch, 32, 40, 174]
         ↓
    Reshape [batch, 174, 640]
         ↓
    RNN (174 timesteps)
         ↓
    FC [batch, 10]
    """
    
    def __init__(self, num_classes=10, input_height=40, input_width=174,
                 hidden_size=128, num_layers=2, dropout_rate=0.5):
        """
        Inicializa o modelo RNN.
        
        Args:
            num_classes (int): Número de categorias (10 para Urban Sound)
            input_height (int): Altura do espectrograma (40 mel bins)
            input_width (int): Largura do espectrograma (174 frames)
            hidden_size (int): Tamanho do estado oculto (padrão: 128)
            num_layers (int): Número de camadas RNN empilhadas (padrão: 2)
            dropout_rate (float): Taxa de dropout (padrão: 0.5)
        """
        super(SoundRNN, self).__init__()
        
        # ========== CAMADA CONVOLUCIONAL ==========
        # Extrai features 2D do espectrograma
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(32)
        # Pool apenas em altura (mantém sequência temporal)
        self.pool_conv = nn.MaxPool2d(kernel_size=(2, 1))
        
        # ========== CAMADA RNN VANILLA ==========
        # Processa sequência temporal
        rnn_input_size = 32 * (input_height // 2)  # 640
        
        # RNN (Vanilla)
        # - nonlinearity: 'tanh' ou 'relu'
        # - tanh é padrão (mais suave)
        # - relu é mais rápido (mas problema com vanishing gradient)
        self.rnn = nn.RNN(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            nonlinearity='tanh'
        )
        
        # ========== CAMADAS FULLY-CONNECTED ==========
        self.fc1 = nn.Linear(hidden_size, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # ========== REGULARIZAÇÃO ==========
        self.dropout = nn.Dropout(dropout_rate)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forward(self, x):
        """
        Forward pass da RNN.
        
        Args:
            x: [batch, 1, 40, 174]
            
        Returns:
            output: [batch, 10]
        """
        batch_size = x.size(0)
        
        # ========== CONVOLUÇÃO ==========
        x = self.conv1(x)              # [batch, 32, 40, 174]
        x = self.bn_conv1(x)
        x = F.relu(x)
        x = self.pool_conv(x)          # [batch, 32, 20, 174]
        
        # ========== RESHAPE PARA SEQUÊNCIA ==========
        x = x.view(batch_size, x.size(3), -1)  # [batch, 174, 640]
        
        # ========== RNN ==========
        # Processa cada frame temporal sequencialmente
        # Forward retorna:
        # - output: todos os hidden states
        # - h_n: último hidden state (o que contém contexto)
        rnn_out, h_n = self.rnn(x)
        # rnn_out: [batch, 174, 128] (todos timesteps)
        # h_n: [num_layers, batch, 128] (último timestep)
        
        # Usar último hidden state (contém contexto da sequência inteira)
        x = h_n[-1]  # [batch, 128]
        
        # ========== CLASSIFICAÇÃO ==========
        x = self.fc1(x)                # [batch, 128]
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)                # [batch, 10]
        
        return x

