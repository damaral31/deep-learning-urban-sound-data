import torch
import torch.nn as nn
import torch.nn.functional as F


class SoundTransformer(nn.Module):
    """
    Transformer para classificação de sons urbanos.
    
    O Transformer é baseado em "attention" e é superior a RNN/LSTM para:
    - Capturar dependências de longo alcance (melhor que RNN)
    - Processar paralelamente (mais rápido que LSTM)
    - Aprender relações complexas entre diferentes partes do áudio
    
    Arquitetura:
    1. Convolução inicial: extrai features 2D do espectrograma
    2. Reshape e projeção: converte para sequência de tokens
    3. Positional encoding: adiciona informação de posição temporal
    4. Transformer encoder: captura relações entre timesteps
    5. Global average pooling: agrega informação de toda sequência
    6. Classificador: predição final
    """
    
    def __init__(self, n_mels=40, n_frames=174, num_classes=10, d_model=128, 
                 nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1, in_channels=1):
        """
        Inicializa o Transformer para áudio.
        
        Args:
            n_mels (int): Número de bandas mel (altura do espectrograma)
            n_frames (int): Número de frames temporais (largura)
            num_classes (int): Número de classes de áudio
            d_model (int): Dimensão de embedding do modelo
            nhead (int): Número de attention heads (deve dividir d_model)
            num_layers (int): Número de camadas transformer
            dim_feedforward (int): Dimensão da camada feedforward interna
            dropout (float): Taxa de dropout
            in_channels (int): Número de canais de entrada (sempre 1 para áudio)
        """
        super(SoundTransformer, self).__init__()
        
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.d_model = d_model
        
        # ========== CAMADA CONVOLUCIONAL INICIAL ==========
        # Extrai features 2D do espectrograma
        # Input: [batch, 1, n_mels, n_frames]
        # Output: [batch, d_model, n_mels, n_frames]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.pool_conv = nn.MaxPool2d(kernel_size=(2, 1))
        
        # Após pooling: altura reduz para n_mels // 2
        self.n_mels_pooled = n_mels // 2
        
        # ========== PROJEÇÃO PARA d_model ==========
        # Converte features 2D em sequência 1D por frame
        # Input por frame: 32 * (n_mels // 2)
        # Output por frame: d_model
        self.feat_proj = nn.Linear(32 * self.n_mels_pooled, d_model)
        
        # ========== POSITIONAL ENCODING ==========
        # Adiciona informação de posição temporal aos embeddings
        # Importante para Transformer saber a ordem dos frames
        # Shape: [1, n_frames, d_model] (broadcast sobre batch)
        self.register_buffer(
            'pos_embedding',
            self._create_positional_encoding(n_frames, d_model)
        )
        
        # ========== TRANSFORMER ENCODER ==========
        # Multi-head self-attention + feedforward por camada
        # Captura relações entre diferentes partes do espectrograma
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # ========== CLASSIFICADOR ==========
        # Camada final para predição
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Dropout final
        self.dropout = nn.Dropout(dropout)

    def _create_positional_encoding(self, seq_length, d_model):
        """
        Cria positional encoding usando sinusóides (estratégia do paper original "Attention is All You Need").
        
        Args:
            seq_length (int): Comprimento da sequência (número de frames)
            d_model (int): Dimensão do modelo
            
        Returns:
            torch.Tensor: Positional encoding [1, seq_length, d_model]
        """
        pos = torch.arange(seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                            -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(1, seq_length, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        
        return pe

    def forward(self, x):
        """
        Forward pass do Transformer.
        
        Args:
            x (torch.Tensor): Input com shape [batch, 1, n_mels, n_frames]
            
        Returns:
            torch.Tensor: Output com shape [batch, num_classes]
        """
        batch_size = x.size(0)
        
        # ========== PASSO 1: CONVOLUÇÃO + BATCH NORM + ReLU ==========
        # Extrai features 2D do espectrograma
        x = self.conv1(x)           # [batch, 32, n_mels, n_frames]
        x = self.bn_conv1(x)
        x = F.relu(x)
        
        # ========== PASSO 2: MAX POOLING (altura) ==========
        x = self.pool_conv(x)       # [batch, 32, n_mels//2, n_frames]
        
        # ========== PASSO 3: RESHAPE EM SEQUÊNCIA ==========
        # Converte de [batch, 32, n_mels//2, n_frames]
        # Para [batch, n_frames, 32*n_mels//2]
        # Cada frame = um token na sequência
        
        _, channels, height, time = x.shape
        # Permute: [batch, 32, n_mels//2, n_frames] → [batch, n_frames, 32, n_mels//2]
        x = x.permute(0, 3, 1, 2)
        # View: [batch, n_frames, 32*n_mels//2]
        x = x.contiguous().view(batch_size, time, channels * height)
        
        # ========== PASSO 4: PROJEÇÃO PARA d_model ==========
        # Converte features para dimensão d_model
        x = self.feat_proj(x)       # [batch, n_frames, d_model]
        
        # ========== PASSO 5: ADICIONAR POSITIONAL ENCODING ==========
        # Informa ao modelo a posição de cada frame
        x = x + self.pos_embedding[:, :x.size(1), :]  # [batch, n_frames, d_model]
        
        # ========== PASSO 6: APLICAR DROPOUT ==========
        x = self.dropout(x)
        
        # ========== PASSO 7: TRANSFORMER ENCODER ==========
        # Multi-head attention: cada head aprende diferentes relações
        # Feedforward: não-linearidade adicional
        x = self.transformer_encoder(x)  # [batch, n_frames, d_model]
        
        # ========== PASSO 8: GLOBAL AVERAGE POOLING ==========
        # Agrega informação de toda a sequência
        # Média sobre dimensão temporal (frames)
        x = x.mean(dim=1)           # [batch, d_model]
        
        # ========== PASSO 9: CLASSIFICADOR ==========
        x = self.classifier(x)      # [batch, num_classes]
        
        return x