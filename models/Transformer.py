import torch
import torch.nn as nn
import torch.nn.functional as F

class SoundTransformer(nn.Module):
    def __init__(self, n_mels=128, n_frames=174, num_classes=10, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(SoundTransformer, self).__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.d_model = d_model

        # Project input to d_model
        self.input_proj = nn.Conv2d(1, d_model, kernel_size=1)
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, n_frames, d_model))
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, 1, n_mels, n_frames)
        x = self.input_proj(x)  # (batch, d_model, n_mels, n_frames)
        x = x.mean(dim=2)       # (batch, d_model, n_frames) - mean over mel bands
        x = x.permute(0, 2, 1)  # (batch, n_frames, d_model)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer_encoder(x)  # (batch, n_frames, d_model)
        x = x.mean(dim=1)       # (batch, d_model) - global average pooling over frames
        x = self.classifier(x)  # (batch, num_classes)
        return x
