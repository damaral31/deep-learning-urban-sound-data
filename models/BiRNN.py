import torch
import torch.nn as nn
import torch.nn.functional as F


class SoundBiRNN(nn.Module):
    def __init__(self, num_classes=10, input_height=40, input_width=174,
                 hidden_size=128, num_layers=2, dropout_rate=0.5, in_channels=1):
        super(SoundBiRNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.pool_conv = nn.MaxPool2d(kernel_size=(2, 1))
        
        # Calculate RNN input size based on architecture
        # Conv2d with padding=1 keeps spatial dims the same
        # MaxPool2d with kernel=(2,1) reduces height by 2
        pooled_height = input_height // 2
        rnn_input_size = 32 * pooled_height
        
        # Bidirectional GRU
        self.birnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # For bidirectional: output is [forward_hidden, backward_hidden]
        # So we concatenate them: hidden_size * 2
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        batch_size = x.size(0)
        
        # Conv and pooling
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.relu(x)
        x = self.pool_conv(x)
        
        # x shape: [batch, 32, height//2, time]
        _, channels, height, time = x.shape
        rnn_input_size = channels * height
        
        # Reshape for RNN: [batch, time_steps, features]
        x = x.permute(0, 3, 1, 2)  # [batch, time, channels, height]
        x = x.contiguous().view(batch_size, time, rnn_input_size)
        
        # Bidirectional GRU forward
        # h_n shape for bidirectional: [num_layers * num_directions, batch, hidden_size]
        # So h_n has shape [num_layers * 2, batch, hidden_size]
        birnn_out, h_n = self.birnn(x)
        
        # Extract final hidden states from both directions
        # h_n[-2] is forward direction of last layer
        # h_n[-1] is backward direction of last layer
        forward_h = h_n[-2]  # [batch, hidden_size]
        backward_h = h_n[-1]  # [batch, hidden_size]
        
        # Concatenate both directions
        context = torch.cat([forward_h, backward_h], dim=1)  # [batch, hidden_size * 2]
        
        # Fully connected layers
        x = self.fc1(context)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x