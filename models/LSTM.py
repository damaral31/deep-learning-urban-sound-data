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
    def __init__(self, num_classes=10, input_height=64, input_width=173,
                 hidden_size=128, num_layers=2, dropout_rate=0.5, in_channels=1):
        super(SoundLSTM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.pool_conv = nn.MaxPool2d(kernel_size=(2, 1))
        # Dummy forward para calcular lstm_input_size
        dummy = torch.zeros(1, in_channels, input_height, input_width)
        x = self.conv1(dummy)
        x = self.bn_conv1(x)
        x = F.relu(x)
        x = self.pool_conv(x)
        _, channels, height, time = x.shape
        lstm_input_size = channels * height
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.relu(x)
        x = self.pool_conv(x)
        _, channels, height, time = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch_size, time, channels * height)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x