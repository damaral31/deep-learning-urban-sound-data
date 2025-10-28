"""
Módulo SoundCNN - Rede Neural Convolucional para classificação de áudio urbano

Este módulo implementa um modelo CNN que processa espectrogramas (imagens de áudio)
para classificar diferentes tipos de sons em ambientes urbanos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoundCNN(nn.Module):
    """
    Rede Neural Convolucional (CNN) para classificação de sons urbanos.
    
    A rede processa espectrogramas 2D (representações visuais de áudio) através de:
    1. Camadas convolucionais: extraem características locais do áudio
    2. Max pooling: reduz o tamanho da imagem
    3. Camadas fully-connected: classificam as características extraídas
    
    Fluxo geral:
    Input (40x174) → Conv → Conv → Conv → Flatten → FC → Output (10 classes)
    """
    
    def __init__(self, num_classes=10):
        """
        Inicializa a arquitetura da rede.
        
        Args:
            num_classes (int): Número de categorias de áudio a classificar (padrão: 10)
                              Ex: 10 sons diferentes de tráfego, sirene, música, etc.
        """
        # Chama o construtor da classe pai (nn.Module)
        super(SoundCNN, self).__init__()
        
        # ========== CAMADAS CONVOLUCIONAIS ==========
        # As convoluções extraem características locais do espectrograma
        # Progressão: 1 → 32 → 64 → 128 (mais filtros = mais características)
        
        # Conv1: Primeira convolução
        # - in_channels=1: espectrograma é monocanal (como uma imagem em escala de cinza)
        # - out_channels=32: extrair 32 características diferentes
        # - kernel_size=3: usa matriz 3x3 para detectar padrões
        # - padding=1: adiciona borda para manter tamanho original
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        # Conv2: Segunda convolução (recebe 32 canais da conv1)
        # - in_channels=32: entrada da conv1
        # - out_channels=64: extrai 64 características combinadas
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Conv3: Terceira convolução (recebe 64 canais da conv2)
        # - in_channels=64: entrada da conv2
        # - out_channels=128: extrai 128 características de alto nível
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # ========== MAX POOLING ==========
        # MaxPool reduz o tamanho da imagem tomando o valor máximo em cada região
        # Benefícios:
        # 1. Reduz tamanho: divide altura e largura por 2
        # 2. Captura features mais importantes (o máximo de cada região)
        # 3. Diminui tamanho computacional e parâmetros
        # 
        # Exemplo com MaxPool(2,2):
        # Input 4x4 → Output 2x2 (divide por 2)
        # Se aplicado 3 vezes: 40x174 → 20x87 → 10x43 → 5x21
        self.pool = nn.MaxPool2d(2, 2)
        
        # ========== DROPOUT (Regularização) ==========
        # Dropout é uma técnica para evitar "overfitting" (memorizar dados)
        # Durante treinamento: desativa aleatoriamente 30% dos neurônios
        # Durante teste: usa todos os neurônios
        # 
        # Por quê? Se a rede memoriza tudo, funciona bem no treino mas falha em dados novos.
        # Dropout força a rede a aprender representações mais robustas.
        # Taxa 0.3 = 30% de chance cada neurônio ser desativado
        self.dropout = nn.Dropout(0.3)
        
        # ========== CAMADAS FULLY-CONNECTED (FC) ==========
        # Depois de extrair características com convoluções,
        # usamos camadas densas para classificar.
        
        # fc1: Primeira camada densa
        # - input: 128 * 5 * 21 = 13440 neurônios
        #   Explicação: após 3 MaxPools (40→20→10→5, 174→87→43→21), 
        #   temos 128 canais de características
        # - output: 256 neurônios (reduz dimensionalidade)
        # 
        # NOTA: Este tamanho é específico para espectrogramas 40x174
        # Se mudar tamanho input, precisa recalcular 128 * altura/8 * largura/8
        self.fc1 = nn.Linear(128 * 5 * 21, 256)  # ajuste conforme tamanho do input
        
        # fc2: Segunda camada densa (camada de saída)
        # - input: 256 neurônios (da fc1)
        # - output: num_classes neurônios (10 para 10 categorias de som)
        # Cada neurônio de saída = confiança em uma classe
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Define o fluxo de dados através da rede (forward pass).
        
        Este método é chamado automaticamente quando fazer:
        output = model(input)
        
        Args:
            x (torch.Tensor): Input com shape [batch_size, 1, altura, largura]
                            Exemplo: [8, 1, 40, 174] = 8 espectrogramas
                            
        Returns:
            torch.Tensor: Output com shape [batch_size, num_classes]
                         Exemplo: [8, 10] = 8 conjuntos de 10 scores (um por classe)
        """
        # ========== BLOCO 1: PRIMEIRA CONVOLUÇÃO + ATIVAÇÃO + POOLING ==========
        # Passo 1: Aplicar primeira convolução
        #   Input:  [batch, 1, 40, 174]
        #   Output: [batch, 32, 40, 174] (tamanho mantido por padding=1)
        x = self.conv1(x)
        
        # Passo 2: Aplicar ReLU (Rectified Linear Unit)
        #   ReLU = max(0, x) → transforma valores negativos em 0
        #   Por quê? Introduz não-linearidade, permite rede aprender padrões complexos
        x = F.relu(x)
        
        # Passo 3: Aplicar MaxPooling
        #   Input:  [batch, 32, 40, 174]
        #   Output: [batch, 32, 20, 87]  (reduz por fator 2)
        x = self.pool(x)
        
        # Resumo Bloco 1: 
        #   Input [batch, 1, 40, 174] → Output [batch, 32, 20, 87]
        
        # ========== BLOCO 2: SEGUNDA CONVOLUÇÃO + ATIVAÇÃO + POOLING ==========
        # Passo 4: Aplicar segunda convolução
        #   Input:  [batch, 32, 20, 87]
        #   Output: [batch, 64, 20, 87]
        x = self.conv2(x)
        
        # Passo 5: ReLU
        x = F.relu(x)
        
        # Passo 6: MaxPooling
        #   Input:  [batch, 64, 20, 87]
        #   Output: [batch, 64, 10, 43]
        x = self.pool(x)
        
        # Resumo Bloco 2: 
        #   Input [batch, 32, 20, 87] → Output [batch, 64, 10, 43]
        
        # ========== BLOCO 3: TERCEIRA CONVOLUÇÃO + ATIVAÇÃO + POOLING ==========
        # Passo 7: Aplicar terceira convolução
        #   Input:  [batch, 64, 10, 43]
        #   Output: [batch, 128, 10, 43]
        x = self.conv3(x)
        
        # Passo 8: ReLU
        x = F.relu(x)
        
        # Passo 9: MaxPooling
        #   Input:  [batch, 128, 10, 43]
        #   Output: [batch, 128, 5, 21]  ← Este é o tamanho usado em fc1!
        x = self.pool(x)
        
        # Resumo Bloco 3: 
        #   Input [batch, 64, 10, 43] → Output [batch, 128, 5, 21]
        
        # ========== FLATTEN (Achatar) ==========
        # Converte tensor 4D em 2D para passar para camadas FC
        # 
        # Antes (4D):  [batch, 128, 5, 21]
        # Depois (2D): [batch, 128*5*21] = [batch, 13440]
        #
        # x.size(0) = tamanho do batch (ex: 8)
        # -1 = calcular automaticamente (128*5*21 = 13440)
        # Assim fica genérico para qualquer batch size!
        x = x.view(x.size(0), -1)
        
        # ========== CAMADAS FULLY-CONNECTED COM REGULARIZAÇÃO ==========
        # Passo 10: Primeira camada densa
        #   Input:  [batch, 13440]
        #   Output: [batch, 256]
        x = self.fc1(x)
        
        # Passo 11: ReLU
        #   Introduz não-linearidade novamente nas camadas densas
        x = F.relu(x)
        
        # Passo 12: Dropout
        #   Durante treinamento: desativa ~30% dos 256 neurônios aleatoriamente
        #   Durante teste: passa todos através (com escala ajustada)
        #   Ajuda a evitar que a rede decore os dados
        x = self.dropout(x)
        
        # Passo 13: Segunda camada densa (saída final)
        #   Input:  [batch, 256]
        #   Output: [batch, 10]  (um score para cada classe)
        #
        # IMPORTANTE: SEM ReLU aqui!
        # Queremos scores em qualquer range, não limitados a [0, ∞)
        # CrossEntropyLoss vai aplicar softmax (converte scores em probabilidades)
        x = self.fc2(x)
        
        # ========== RESUMO DO FLUXO COMPLETO ==========
        # Input:  [batch, 1, 40, 174]  ← Espectrograma
        #   ↓
        # Conv1+ReLU+Pool: [batch, 32, 20, 87]
        #   ↓
        # Conv2+ReLU+Pool: [batch, 64, 10, 43]
        #   ↓
        # Conv3+ReLU+Pool: [batch, 128, 5, 21]
        #   ↓
        # Flatten: [batch, 13440]
        #   ↓
        # FC1+ReLU+Dropout: [batch, 256]
        #   ↓
        # FC2: [batch, 10]  ← Scores finais (logits)
        #   ↓
        # (em seguida: softmax + argmax para obter classe final)
        
        return x