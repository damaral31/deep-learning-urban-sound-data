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


class SoundBiRNN(nn.Module):
    """
    RNN Bidirecional para classificação de sons urbanos.
    
    Bidirecional significa:
    - Forward: processa da esquerda para direita (frame 0 → 174)
    - Backward: processa da direita para esquerda (frame 174 → 0)
    - Resultado: concatenação (contexto completo)
    
    Vantagem sobre unidirecional:
    - Cada timestep tem contexto do passado E do futuro
    - Melhor para áudio (som no futuro pode influenciar classificação)
    
    Exemplo:
    Sirene: frame inicial pode ser ambíguo, mas com contexto futuro
            sabemos que é sirene (som sobe progressivamente)
    
    Trade-off:
    - Mais parâmetros (2x da versão unidirecional)
    - Mais lento
    - Melhor resultado
    
    Para áudio urbano: RECOMENDADO!
    """
    
    def __init__(self, num_classes=10, input_height=40, input_width=174,
                 hidden_size=128, num_layers=2, dropout_rate=0.5):
        """Inicializa RNN Bidirecional."""
        super(SoundBiRNN, self).__init__()
        
        # ========== CAMADA CONVOLUCIONAL ==========
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.pool_conv = nn.MaxPool2d(kernel_size=(2, 1))
        
        # ========== CAMADA RNN BIDIRECIONAL ==========
        rnn_input_size = 32 * (input_height // 2)
        
        self.birnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True  # ← Chave: bidirecional
        )
        
        # Output bidirecional = 2 * hidden_size
        # Forward hidden: 128, Backward hidden: 128
        # Total: 256
        
        # ========== CAMADAS FULLY-CONNECTED ==========
        self.fc1 = nn.Linear(hidden_size * 2, 128)  # * 2 por bidirecional
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # ========== REGULARIZAÇÃO ==========
        self.dropout = nn.Dropout(dropout_rate)
        
        self.hidden_size = hidden_size
    
    def forward(self, x):
        """Forward pass RNN Bidirecional."""
        batch_size = x.size(0)
        
        # Convolução
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.relu(x)
        x = self.pool_conv(x)
        
        # Reshape
        x = x.view(batch_size, x.size(3), -1)
        
        # RNN Bidirecional
        # h_n shape: [num_layers * 2, batch, hidden_size]
        # (* 2 por bidirecional: forward + backward)
        birnn_out, h_n = self.birnn(x)
        
        # Combinar forward e backward do último layer
        # Forward: h_n[-2]  (últimoframe, direção forward)
        # Backward: h_n[-1] (primeiro frame, direção backward)
        forward_h = h_n[-2]      # [batch, 128]
        backward_h = h_n[-1]     # [batch, 128]
        context = torch.cat([forward_h, backward_h], dim=1)  # [batch, 256]
        
        # FC layers
        x = self.fc1(context)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ============================================================================
# EXEMPLOS DE USO E TESTES
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("MODELOS RNN PARA CLASSIFICAÇÃO DE ÁUDIO URBANO")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📱 Usando device: {device}")
    
    # Parâmetros
    batch_size = 8
    num_classes = 10
    input_height = 40
    input_width = 174
    
    # Input simulado
    dummy_input = torch.randn(batch_size, 1, input_height, input_width).to(device)
    print(f"\n📊 Input shape: {dummy_input.shape}")
    print(f"   {batch_size} espectrogramas × {input_height} mel bins × {input_width} frames")
    
    # ===== MODELO 1: RNN VANILLA =====
    print("\n" + "=" * 80)
    print("MODELO 1: SoundRNN (RNN Vanilla)")
    print("=" * 80)
    
    rnn_model = SoundRNN(num_classes=num_classes).to(device)
    rnn_params = sum(p.numel() for p in rnn_model.parameters())
    print(f"📈 Parâmetros: {rnn_params:,}")
    
    with torch.no_grad():
        rnn_out = rnn_model(dummy_input)
    print(f"✅ Output: {rnn_out.shape}")
    print(f"   Classe predita: {rnn_out.argmax(dim=1).cpu().numpy()}")
    
    # ===== MODELO 2: GRU =====
    print("\n" + "=" * 80)
    print("MODELO 2: SoundGRU (Gated Recurrent Unit)")
    print("=" * 80)
    
    gru_model = SoundGRU(num_classes=num_classes).to(device)
    gru_params = sum(p.numel() for p in gru_model.parameters())
    print(f"📈 Parâmetros: {gru_params:,}")
    
    with torch.no_grad():
        gru_out = gru_model(dummy_input)
    print(f"✅ Output: {gru_out.shape}")
    print(f"   Classe predita: {gru_out.argmax(dim=1).cpu().numpy()}")
    
    # ===== MODELO 3: RNN BIDIRECIONAL =====
    print("\n" + "=" * 80)
    print("MODELO 3: SoundBiRNN (RNN Bidirecional)")
    print("=" * 80)
    
    birnn_model = SoundBiRNN(num_classes=num_classes).to(device)
    birnn_params = sum(p.numel() for p in birnn_model.parameters())
    print(f"📈 Parâmetros: {birnn_params:,}")
    
    with torch.no_grad():
        birnn_out = birnn_model(dummy_input)
    print(f"✅ Output: {birnn_out.shape}")
    print(f"   Classe predita: {birnn_out.argmax(dim=1).cpu().numpy()}")
    
    # ===== COMPARAÇÃO =====
    print("\n" + "=" * 80)
    print("COMPARAÇÃO DOS MODELOS")
    print("=" * 80)
    print(f"""
Modelo              Parâmetros    Velocidade   Melhor Para
────────────────────────────────────────────────────────────────
RNN Vanilla         {rnn_params:>12,}    ⚡ Rápida     Sequências curtas
GRU                 {gru_params:>12,}    ⚡⚡ Média      Equilíbrio (RECOMENDADO)
RNN Bidirecional    {birnn_params:>12,}    ⚡ Lenta      Contexto completo

Urban Sound (10 classes):
  0: Air conditioner      5: Engine idling
  1: Car horn             6: Gun shot
  2: Children playing     7: Jackhammer
  3: Dog barking          8: Siren
  4: Drilling             9: Street music

Recomendação: Use GRU (melhor equilíbrio) ou BiRNN (melhor precisão)
    """)
    
    print("=" * 80)
    print("✅ Testes completados com sucesso!")
    print("=" * 80)
