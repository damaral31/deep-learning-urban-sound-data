import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# Attention Block (Channel Attention)
class ChannelAttentionBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        out = out.view(b, c, 1, 1)
        return x * out.expand_as(x)

class SoundCNN(nn.Module):
    def __init__(self, num_classes=10, SqueezeExcitation=False, AttentionBlock=False, in_channels=1):
        super(SoundCNN, self).__init__()
        self.SqueezeExcitation = SqueezeExcitation
        self.AttentionBlock = AttentionBlock

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        if SqueezeExcitation:
            self.se = SEBlock(128)
        if AttentionBlock:
            self.attn = ChannelAttentionBlock(128)

        # Adaptive pooling to handle variable input sizes
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1))
        if self.SqueezeExcitation:
            x = self.se(x)
        if self.AttentionBlock:
            x = self.attn(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.leaky_relu(self.fc1(x), negative_slope=0.1))
        x = self.fc2(x)
        return x