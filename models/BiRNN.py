import torch
import torch.nn as nn
import torch.nn.functional as F

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
