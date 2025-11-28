import torch
import torch.nn as nn
import torch.nn.functional as F


class SoundRNN(nn.Module):
    def __init__(self, num_classes=10, input_height=64, input_width=173,
                 hidden_size=128, num_layers=2, dropout_rate=0.5, in_channels=1):
        super(SoundRNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.pool_conv = nn.MaxPool2d(kernel_size=(2, 1))
        
        # Calculate actual dimensions after conv and pooling
        # Conv2d with padding=1 keeps spatial dims the same
        # MaxPool2d with kernel=(2,1) reduces height by 2
        pooled_height = input_height // 2
        pooled_width = input_width  # MaxPool kernel is (2,1), so width unchanged
        
        # RNN input size = channels * height after pooling
        rnn_input_size = 32 * pooled_height
        
        self.rnn = nn.RNN(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            nonlinearity='tanh'
        )
        
        self.fc1 = nn.Linear(hidden_size, 128)
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
        
        # x shape: [batch, 32, height//2, width]
        _, channels, height, time = x.shape
        
        # Reshape for RNN: [batch, time_steps, features]
        # We treat time dimension as sequence, and (channels * height) as features
        x = x.permute(0, 3, 1, 2)  # [batch, time, channels, height]
        x = x.contiguous().view(batch_size, time, channels * height)
        
        # RNN forward
        rnn_out, h_n = self.rnn(x)
        
        # Use final hidden state
        x = h_n[-1]  # [batch, hidden_size]
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x