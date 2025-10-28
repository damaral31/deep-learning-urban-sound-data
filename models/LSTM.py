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
    
    Fluxo de dados:
    Input [batch, 1, 40, 174] (espectrograma)
         ↓
    Conv [batch, 32, 40, 174] (features 2D)
         ↓
    Reshape [batch, 174, 32*40] (sequência temporal)
         ↓
    LSTM Bidirecional [batch, 174, 256] (contexto temporal)
         ↓
    FC layers [batch, 10] (classificação)
    """
    
    def __init__(self, num_classes=10, input_height=40, input_width=174,
                 hidden_size=128, num_layers=2, dropout_rate=0.5):
        """
        Inicializa o modelo LSTM.
        
        Args:
            num_classes (int): Número de categorias de áudio (padrão: 10)
                              Classes típicas Urban Sound: 
                              - 0: Air conditioner
                              - 1: Car horn
                              - 2: Children playing
                              - 3: Dog barking
                              - 4: Drilling
                              - 5: Engine idling
                              - 6: Gun shot
                              - 7: Jackhammer
                              - 8: Siren
                              - 9: Street music
            
            input_height (int): Altura do espectrograma (padrão: 40 - mel bins)
                               Tipicamente 40 para espectrogramas de mel
            
            input_width (int): Largura do espectrograma (padrão: 174 - frames)
                              Corresponde ao número de frames temporais
            
            hidden_size (int): Dimensão do estado oculto LSTM (padrão: 128)
                              Quanto maior = mais capaz de aprender, mas mais lento
            
            num_layers (int): Número de camadas LSTM empilhadas (padrão: 2)
                             Mais camadas = mais profundo, mas risco de overfitting
            
            dropout_rate (float): Taxa de dropout para regularização (padrão: 0.5)
        """
        super(SoundLSTM, self).__init__()
        
        # ========== CAMADA CONVOLUCIONAL INICIAL ==========
        # Propósito: Extrair features 2D do espectrograma
        # Uma convolução pode ser suficiente para pré-processar
        # (diferente de CNN que usa 3+ camadas)
        
        # Conv1: Primeira e única convolução
        # - in_channels=1: espectrograma é monocanal
        # - out_channels=32: extrai 32 características
        # - kernel_size=3: filtro 3x3 para padrões locais
        # - padding=1: mantém o tamanho
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        # BatchNorm2d: normaliza outputs da convolução
        # Benefícios:
        # - Treinamento mais estável
        # - Permite maiores learning rates
        # - Reduz dependência de inicialização de pesos
        self.bn_conv1 = nn.BatchNorm2d(32)
        
        # ReLU: ativação não-linear
        # Já definida no forward, mas poderia ser aqui também
        
        # MaxPool: reduz altura (largura mantida para sequência temporal)
        # - kernel_size=(2, 1): reduz altura por 2, não mexe na largura
        # - Por quê? Queremos manter a dimensão temporal (largura) para LSTM
        # Após pool: 40→20 (altura), 174→174 (largura)
        self.pool_conv = nn.MaxPool2d(kernel_size=(2, 1))
        
        # ========== CAMADA LSTM BIDIRECIONAL ==========
        # Propósito: Capturar dependências temporais em ambas direções
        
        # Cálculo do tamanho de entrada LSTM:
        # Após convução + pooling: [batch, 32 canais, 20 altura, 174 largura]
        # Queremos tratar a largura como sequência temporal
        # Cada timestep = 32 * 20 = 640 features
        lstm_input_size = 32 * (input_height // 2)  # 32 * 20 = 640
        
        # LSTM bidirecional
        # - input_size: 640 (features de entrada por timestep)
        # - hidden_size: 128 (tamanho estado oculto)
        # - num_layers: 2 (2 camadas LSTM empilhadas)
        # - bidirectional: True (processa em ambas direções)
        #   * Forward: 0→1→2→...→174
        #   * Backward: 174→...→2→1→0
        #   * Output concatenado: [forward, backward] = 256 features
        # - batch_first: True (input shape: [batch, seq, features])
        # - dropout: 0.5 (dropout entre camadas, reduz overfitting)
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # ========== CAMADAS FULLY-CONNECTED ==========
        # Depois de processar sequência temporal com LSTM,
        # usamos FC layers para classificação
        
        # Nota: hidden_size * 2 porque LSTM é bidirecional
        # Forward + Backward = 128 + 128 = 256 features
        
        # fc1: Primeira camada densa com dropout
        # - input: hidden_size * 2 = 256 (output bidirecional LSTM)
        # - output: 128 (reduz dimensionalidade)
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        
        # BatchNorm1d: normaliza outputs da fc1
        # Mesmo conceito que BatchNorm2d, mas para tensores 1D
        self.bn_fc1 = nn.BatchNorm1d(128)
        
        # fc2: Segunda camada densa (camada de saída)
        # - input: 128 (da fc1)
        # - output: num_classes = 10 (classificação final)
        self.fc2 = nn.Linear(128, num_classes)
        
        # ========== REGULARIZAÇÃO ==========
        # Dropout aplicado entre FC layers durante treinamento
        # Taxa: 0.5 = 50% dos neurônios desativados aleatoriamente
        self.dropout = nn.Dropout(dropout_rate)
        
        # Guardar hiperparâmetros para referência
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

    def forward(self, x):
        """
        Define o fluxo de dados através da rede (forward pass).
        
        Este método é chamado automaticamente quando fazer:
        output = model(input)
        
        Args:
            x (torch.Tensor): Input espectrograma com shape [batch_size, 1, altura, largura]
                            Exemplo: [8, 1, 40, 174]
                            8 espectrogramas de áudio urbano
                            
        Returns:
            torch.Tensor: Logits de saída com shape [batch_size, num_classes]
                         Exemplo: [8, 10]
                         8 conjuntos de 10 scores (um por classe de som)
        """
        batch_size = x.size(0)
        
        # ========== PASSO 1: CONVOLUÇÃO + BATCH NORM + ReLU ==========
        # Propósito: Extrair features 2D do espectrograma
        #
        # Input shape: [batch, 1, 40, 174]
        # Esperado output: [batch, 32, 40, 174] (sem pool ainda)
        
        # Aplicar convolução
        x = self.conv1(x)  # [batch, 32, 40, 174]
        
        # Batch normalization: normaliza outputs
        x = self.bn_conv1(x)
        
        # ReLU: ativação não-linear
        # max(0, x) → apenas valores positivos
        x = F.relu(x)
        
        # ========== PASSO 2: MAX POOLING (ALTURA) ==========
        # Reduz altura por 2, mantém largura
        # [batch, 32, 40, 174] → [batch, 32, 20, 174]
        x = self.pool_conv(x)
        
        # ========== PASSO 3: RESHAPE PARA SEQUÊNCIA TEMPORAL ==========
        # Converte tensor 4D em 3D para LSTM
        #
        # Antes: [batch, 32, 20, 174]  (4D: batch, canais, altura, largura)
        # Depois: [batch, 174, 640]     (3D: batch, timesteps, features)
        #
        # Interpretação:
        # - Dimension temporal (174 timesteps) = cada coluna do espectrograma
        # - Features (640) = 32 canais × 20 altura
        # - Cada timestep representa um frame temporal do áudio
        
        # Dimensões:
        # x.size(0) = batch size
        # x.size(3) = 174 (largura = número de frames)
        # -1 = calcular automaticamente (32 * 20 = 640)
        
        x = x.view(batch_size, x.size(3), -1)
        # Agora: [batch, 174, 640]
        
        # ========== PASSO 4: LSTM BIDIRECIONAL ==========
        # Propósito: Capturar contexto temporal em ambas direções
        #
        # LSTM processa sequência e retorna:
        # - lstm_out: output de todos os timesteps
        # - (h_n, c_n): estado final (hidden e cell state)
        #
        # Processamento bidirecional:
        # Forward:  Frame0 → Frame1 → Frame2 → ... → Frame173
        # Backward: Frame173 → ... → Frame2 → Frame1 → Frame0
        # Result: Concatenação de ambas = contexto completo
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: [batch, 174, 256]
        #                 (256 = 128 hidden_size * 2 directions)
        # h_n shape: [num_layers * 2, batch, 128]  (* 2 por bidirecional)
        # c_n shape: [num_layers * 2, batch, 128]
        
        # ========== PASSO 5: POOL TEMPORAL ==========
        # Strategy: Usar apenas o último output LSTM
        # Justificativa:
        # - Contém contexto de toda a sequência (LSTM propaga contexto)
        # - Mais estável que média
        # - Alternativa: usar média de todos os timesteps ou attention
        #
        # Outro estratégia seria:
        # x = lstm_out.mean(dim=1)  # Média de todos timesteps
        #
        # Ou usar atenção para aprender pesos dinamicamente
        
        last_output = lstm_out[:, -1, :]
        # Shape: [batch, 256]
        # -1 seleciona último timestep
        
        # ========== PASSO 6: PRIMEIRA CAMADA FULLY-CONNECTED ==========
        # Reduz dimensionalidade e aprende combinações não-lineares
        
        x = self.fc1(last_output)  # [batch, 128]
        
        # Batch normalization: estabiliza treinamento
        x = self.bn_fc1(x)
        
        # ReLU: ativação não-linear
        x = F.relu(x)
        
        # Dropout: regularização para evitar overfitting
        x = self.dropout(x)
        
        # ========== PASSO 7: SEGUNDA CAMADA FULLY-CONNECTED (SAÍDA) ==========
        # Camada final: produz scores para cada classe
        #
        # IMPORTANTE: SEM ativação aqui!
        # Retornamos logits (scores brutos)
        # CrossEntropyLoss vai aplicar softmax internamente
        
        x = self.fc2(x)  # [batch, 10]
        
        # ========== RESUMO DO FLUXO COMPLETO ==========
        # [batch, 1, 40, 174]        ← Espectrograma de áudio urbano
        #   ↓
        # Conv + BN + ReLU + Pool
        # [batch, 32, 20, 174]       ← Features 2D extraídas
        #   ↓
        # Reshape para sequência
        # [batch, 174, 640]          ← 174 timesteps com 640 features cada
        #   ↓
        # LSTM Bidirecional
        # [batch, 256]               ← Contexto temporal (último timestep)
        #   ↓
        # FC1 + BN + ReLU + Dropout
        # [batch, 128]               ← Features aprendidas
        #   ↓
        # FC2
        # [batch, 10]                ← Scores finais (logits)
        #   ↓
        # Softmax (durante inference)
        # [batch, 10]                ← Probabilidades (soma = 1)
        #   ↓
        # Argmax
        # [batch]                    ← Classe final (0-9)
        
        return x


class SoundLSTMAttention(nn.Module):
    """
    Versão melhorada de SoundLSTM com Attention Mechanism.
    
    Melhoria:
    - Ao invés de usar apenas o último output LSTM,
    - Usa attention para aprender quais timesteps são mais importantes
    - Melhor que pooling simples, especialmente para sounds curtos ou longos
    
    Uso: Mesma interface que SoundLSTM
    """
    
    def __init__(self, num_classes=10, input_height=40, input_width=174,
                 hidden_size=128, num_layers=2, dropout_rate=0.5):
        """Inicializa LSTM com Attention."""
        super(SoundLSTMAttention, self).__init__()
        
        # Mesmas camadas da versão anterior
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.pool_conv = nn.MaxPool2d(kernel_size=(2, 1))
        
        lstm_input_size = 32 * (input_height // 2)
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # ========== ATTENTION MECHANISM ==========
        # Aprende quais timesteps são mais relevantes
        
        # Linear layer para calcular scores de atenção
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # FC layers
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_size = hidden_size
    
    def forward(self, x):
        """Forward pass com attention."""
        batch_size = x.size(0)
        
        # Convolução
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.relu(x)
        x = self.pool_conv(x)
        
        # Reshape
        x = x.view(batch_size, x.size(3), -1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        # [batch, 174, 256]
        
        # ========== ATTENTION ==========
        # Calcular scores de atenção para cada timestep
        attention_scores = self.attention(lstm_out)
        # [batch, 174, 1]
        
        # Aplicar softmax para normalizar scores
        attention_weights = F.softmax(attention_scores, dim=1)
        # [batch, 174, 1]
        
        # Ponderar outputs LSTM pelo peso de atenção
        weighted_output = lstm_out * attention_weights
        # [batch, 174, 256] * [batch, 174, 1] = [batch, 174, 256]
        
        # Somar ponderado (reduz timesteps)
        context = weighted_output.sum(dim=1)
        # [batch, 256]
        
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
    print("=" * 70)
    print("MODELOS LSTM PARA CLASSIFICAÇÃO DE ÁUDIO URBANO")
    print("=" * 70)
    
    # ===== CONFIGURAÇÃO =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📱 Usando device: {device}")
    
    # Parâmetros
    batch_size = 8
    num_classes = 10  # Urban Sound: 10 categorias
    input_height = 40  # Mel bins
    input_width = 174  # Frames temporais
    
    # Criar input simulado (batch de 8 espectrogramas)
    dummy_input = torch.randn(batch_size, 1, input_height, input_width).to(device)
    print(f"\n📊 Input shape: {dummy_input.shape}")
    print(f"   Interpretação: {batch_size} espectrogramas, "
          f"{input_height} mel bins, {input_width} frames")
    
    # ===== MODELO 1: LSTM BÁSICO =====
    print("\n" + "=" * 70)
    print("MODELO 1: SoundLSTM (LSTM Bidirecional)")
    print("=" * 70)
    
    lstm_model = SoundLSTM(
        num_classes=num_classes,
        input_height=input_height,
        input_width=input_width,
        hidden_size=128,
        num_layers=2,
        dropout_rate=0.5
    ).to(device)
    
    # Contar parâmetros
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    print(f"\n📈 Total de parâmetros: {lstm_params:,}")
    
    # Forward pass
    with torch.no_grad():
        lstm_output = lstm_model(dummy_input)
    
    print(f"✅ Output shape: {lstm_output.shape}")
    print(f"   Interpretação: {batch_size} amostras, {num_classes} classes")
    
    # Calcular probabilidades (softmax)
    probs = torch.softmax(lstm_output, dim=1)
    predicted_classes = torch.argmax(probs, dim=1)
    print(f"\n📋 Exemplo de predição:")
    print(f"   Scores brutos: {lstm_output[0]}")
    print(f"   Probabilidades: {probs[0]}")
    print(f"   Classe predita: {predicted_classes[0].item()}")
    
    # ===== MODELO 2: LSTM COM ATTENTION =====
    print("\n" + "=" * 70)
    print("MODELO 2: SoundLSTMAttention (LSTM + Attention)")
    print("=" * 70)
    
    attention_model = SoundLSTMAttention(
        num_classes=num_classes,
        input_height=input_height,
        input_width=input_width,
        hidden_size=128,
        num_layers=2,
        dropout_rate=0.5
    ).to(device)
    
    # Contar parâmetros
    attention_params = sum(p.numel() for p in attention_model.parameters())
    print(f"\n📈 Total de parâmetros: {attention_params:,}")
    
    # Forward pass
    with torch.no_grad():
        attention_output = attention_model(dummy_input)
    
    print(f"✅ Output shape: {attention_output.shape}")
    
    # ===== COMPARAÇÃO =====
    print("\n" + "=" * 70)
    print("COMPARAÇÃO")
    print("=" * 70)
    print(f"SoundLSTM:           {lstm_params:,} parâmetros")
    print(f"SoundLSTMAttention:  {attention_params:,} parâmetros")
    print(f"\nDiferença: {attention_params - lstm_params:,} parâmetros")
    print(f"Razão: {attention_params / lstm_params:.2f}x")
    
    # ===== INFORMAÇÕES TÉCNICAS =====
    print("\n" + "=" * 70)
    print("INFORMAÇÕES TÉCNICAS")
    print("=" * 70)
    print(f"""
Urban Sound Dataset (10 classes):
  0: Air conditioner      6: Gun shot
  1: Car horn             7: Jackhammer
  2: Children playing     8: Siren
  3: Dog barking          9: Street music
  4: Drilling
  5: Engine idling

Características LSTM para áudio urbano:
  ✓ Captura dependências temporais
  ✓ Bidirecional: contexto anterior e posterior
  ✓ Melhor que CNN para sequências longas
  ✓ Attention: aprende frames importantes

Tamanho típico espectrograma:
  - Altura: 40 (mel-frequency bins)
  - Largura: 174 (frames temporais)
  - Duração: ~4 segundos (44.1 kHz, hop=512)
    """)
    
    print("\n" + "=" * 70)
    print("✅ Testes completados com sucesso!")
    print("=" * 70)
