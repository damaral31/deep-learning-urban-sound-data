import torch
import torch.nn as nn
import torch.nn.functional as F

class SoundGRU(nn.Module):
    """
    Rede com Gated Recurrent Units (GRU) para classificação de sons urbanos.
    
    GRU é uma versão simplificada de LSTM:
    - Mantém os benefícios de LSTM (evita vanishing gradient)
    - Mas é mais simples e mais rápida
    - Menos parâmetros que LSTM
    
    Diferenças GRU vs LSTM:
    ┌─────────────────────────────────────────────────┐
    │             GRU         │       LSTM              │
    ├─────────────────────────┼───────────────────────┤
    │ Gates:    2             │ 3                     │
    │ Hidden:   1             │ 2 (hidden + cell)     │
    │ Parâm:    3x mais       │ 4x mais (que RNN)    │
    │ Velocidade: Mais rápida │ Mais lenta             │
    │ Sequência: até 1000     │ Até 10000+            │
    └─────────────────────────┴───────────────────────┘
    
    Quando usar GRU vs LSTM:
    - GRU: Menos dados, sequências curtas (<500), treino rápido
    - LSTM: Mais dados, sequências longas, problema completo
    
    Para áudio urbano (~4 segundos): GRU é suficiente!
    """
    
    def __init__(self, num_classes=10, input_height=40, input_width=174,
                 hidden_size=128, num_layers=2, dropout_rate=0.5):
        """
        Inicializa o modelo GRU.
        
        Args:
            num_classes (int): Número de categorias
            input_height (int): Altura do espectrograma
            input_width (int): Largura do espectrograma
            hidden_size (int): Tamanho do estado oculto
            num_layers (int): Número de camadas GRU
            dropout_rate (float): Taxa de dropout
        """
        super(SoundGRU, self).__init__()
        
        # ========== CAMADA CONVOLUCIONAL ==========
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.pool_conv = nn.MaxPool2d(kernel_size=(2, 1))
        
        # ========== CAMADA GRU ==========
        # GRU é similar a RNN, mas com gates para melhor aprendizado
        rnn_input_size = 32 * (input_height // 2)
        
        self.gru = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False  # Unidirecional (podia ser bidirecional também)
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
        """Forward pass da GRU."""
        batch_size = x.size(0)
        
        # Convolução
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.relu(x)
        x = self.pool_conv(x)
        
        # Reshape
        x = x.view(batch_size, x.size(3), -1)
        
        # GRU
        gru_out, h_n = self.gru(x)
        x = h_n[-1]  # Último hidden state
        
        # FC layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
