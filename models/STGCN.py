"""
STGCN - Spatial-Temporal Graph Convolutional Network

Modelo de rede neural que combina:
- Graph Convolutions (captura relações espaciais)
- Temporal Convolutions (captura dependências temporais)

Ideal para dados com estrutura de grafo + dimensão temporal.
Para Urban Sound: Cada frequência (nó) está conectada a frequências vizinhas.

Autor: Deep Learning Urban Sound
Versão: 1.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# ADJACÊNCIA E LAPLACIANA
# ============================================================================

def create_frequency_adjacency(num_freq_bins=40):
    """
    Cria matriz de adjacência conectando frequências vizinhas.
    
    Cada frequência (nó) está conectada à frequência anterior e próxima,
    formando uma linha (chain graph).
    """
    A = np.eye(num_freq_bins)
    
    for i in range(num_freq_bins - 1):
        A[i, i+1] = 1
        A[i+1, i] = 1
    
    return A.astype(np.float32)


def compute_laplacian_normalized(A):
    """Calcula matriz Laplaciana normalizada L̃ = I - D^(-1/2) @ A @ D^(-1/2)"""
    D = np.diag(A.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D.diagonal() + 1e-8))
    I = np.eye(A.shape[0])
    
    L_tilde = I - D_inv_sqrt @ A @ D_inv_sqrt
    
    return L_tilde.astype(np.float32)


# ============================================================================
# GRAPH CONVOLUTION COM CHEBYSHEV POLYNOMIALS
# ============================================================================

class ChebConvolution(nn.Module):
    """
    Graph Convolution usando Chebyshev Polynomials.
    Aproxima convoluções em grafos usando expansão em polinômios de Chebyshev.
    """
    
    def __init__(self, L_tilde, in_channels, out_channels, K=3):
        super().__init__()
        
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.register_buffer('L_tilde', torch.FloatTensor(L_tilde))
        
        self.weights = nn.Parameter(torch.randn(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        nn.init.xavier_uniform_(self.weights)
    
    def forward(self, x):
        """
        Args:
            x: Features dos nós [batch, num_nodes, in_channels]
        
        Returns:
            y: Features convoluídas [batch, num_nodes, out_channels]
        """
        batch_size, num_nodes, in_channels = x.shape
        device = x.device
        
        T = [x]
        
        if self.K > 1:
            T.append(torch.matmul(self.L_tilde.to(device), x))
        
        for k in range(2, self.K):
            T_next = 2 * torch.matmul(self.L_tilde.to(device), T[k-1]) - T[k-2]
            T.append(T_next)
        
        y = torch.zeros(batch_size, num_nodes, self.out_channels, device=device)
        
        for k in range(self.K):
            y += torch.matmul(T[k], self.weights[k])
        
        y = y + self.bias
        
        return y


# ============================================================================
# TEMPORAL CONVOLUTION
# ============================================================================

class TemporalConvolution(nn.Module):
    """Convolução temporal 1D para capturar padrões no tempo."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=True
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, num_nodes, timesteps, in_channels]
        
        Returns:
            y: [batch, num_nodes, timesteps, out_channels]
        """
        batch, num_nodes, timesteps, in_channels = x.shape
        
        x = x.reshape(batch * num_nodes, in_channels, timesteps)
        y = self.conv(x)
        y = y.reshape(batch, num_nodes, y.shape[-1], y.shape[1])
        y = y.permute(0, 1, 2, 3)
        
        return y


# ============================================================================
# ST-CONV BLOCK
# ============================================================================

class STConvBlock(nn.Module):
    """Bloco Spatial-Temporal Convolution completo."""
    
    def __init__(self, L_tilde, in_channels, out_channels, 
                 kernel_size_temporal=3, kernel_size_graph=3, dropout=0.5):
        super().__init__()
        
        self.temporal_conv1 = TemporalConvolution(
            in_channels, out_channels, 
            kernel_size=kernel_size_temporal
        )
        
        self.graph_conv = ChebConvolution(
            L_tilde, out_channels, out_channels,
            K=kernel_size_graph
        )
        
        self.temporal_conv2 = TemporalConvolution(
            out_channels, out_channels,
            kernel_size=kernel_size_temporal
        )
        
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [batch, num_nodes, timesteps, in_channels]
        
        Returns:
            y: [batch, num_nodes, timesteps, out_channels]
        """
        y = self.temporal_conv1(x)
        
        batch, num_nodes, timesteps, out_channels = y.shape
        y_spatial_list = []
        
        for t in range(timesteps):
            y_t = y[:, :, t, :]
            y_t = self.graph_conv(y_t)
            y_spatial_list.append(y_t)
        
        y = torch.stack(y_spatial_list, dim=2)
        
        y = self.temporal_conv2(y)
        
        batch, num_nodes, timesteps, out_channels = y.shape
        y = y.permute(0, 3, 1, 2)
        y = self.batch_norm(y)
        y = y.permute(0, 2, 3, 1)
        
        y = self.relu(y)
        y = self.dropout(y)
        
        return y


# ============================================================================
# STGCN COMPLETO
# ============================================================================

class SoundSTGCN(nn.Module):
    """
    STGCN completo para classificação de Urban Sound 8K.
    
    Arquitetura:
    - Input: [batch, 1, 40, 174]
    - STConvBlock 1: 1 -> 64 channels
    - STConvBlock 2: 64 -> 128 channels
    - STConvBlock 3: 128 -> 256 channels
    - Global Average Pooling
    - Dense layer: 256 -> 10 (classes)
    """
    
    def __init__(self, num_classes=10, num_freq_bins=40, dropout=0.5):
        super().__init__()
        
        A = create_frequency_adjacency(num_freq_bins)
        L_tilde = compute_laplacian_normalized(A)
        
        self.st_block1 = STConvBlock(
            L_tilde, in_channels=1, out_channels=64,
            kernel_size_temporal=3, kernel_size_graph=3,
            dropout=dropout
        )
        
        self.st_block2 = STConvBlock(
            L_tilde, in_channels=64, out_channels=128,
            kernel_size_temporal=3, kernel_size_graph=3,
            dropout=dropout
        )
        
        self.st_block3 = STConvBlock(
            L_tilde, in_channels=128, out_channels=256,
            kernel_size_temporal=3, kernel_size_graph=3,
            dropout=dropout
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        self.dropout_final = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [batch, 1, 40, 174]
        
        Returns:
            logits: [batch, 10]
        """
        batch_size = x.shape[0]
        
        x = x.permute(0, 2, 3, 1)
        
        x = self.st_block1(x)
        x = self.st_block2(x)
        x = self.st_block3(x)
        
        batch, num_nodes, timesteps, channels = x.shape
        x = x.permute(0, 3, 1, 2)
        
        x = self.avg_pool(x)
        x = x.reshape(batch_size, -1)
        
        x = self.dropout_final(x)
        x = self.fc(x)
        
        return x


if __name__ == "__main__":
    print("="*80)
    print("🧪 TESTE STGCN")
    print("="*80)
    
    model = SoundSTGCN(num_classes=10, num_freq_bins=40)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros: {num_params:,}")
    
    x = torch.randn(32, 1, 40, 174)
    y = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print("✅ OK!\n")
