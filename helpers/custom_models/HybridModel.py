import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from helpers.setup import set_seeds

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expansion=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU(inplace=True) # Use Swish activation

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y

class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class HybridModel(nn.Module):
    def __init__(self, device, out_features=39, dropout_proba=0.5):
        super(HybridModel, self).__init__()
        # set device
        self.device = device

        # Convolutional blocks
        self.conv1 = ConvBlock(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.se3 = SEBlock(128)

        # Flatten to prepare for transformer
        self.flatten = nn.Flatten(start_dim=2)
        
        # Attention blocks
        self.attn1 = AttentionBlock(dim=128, num_heads=4)
        self.attn2 = AttentionBlock(dim=128, num_heads=4)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_proba)

        # Classification head
        self.fc = nn.Linear(128 * 28 * 28, out_features)

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.se3(x)  # Apply SEBlock after conv3

        # Prepare for attention
        B, C, H, W = x.shape
        x = self.flatten(x)
        x = x.transpose(1, 2)  # (B, N, C)

        x = self.attn1(x)
        x = self.attn2(x)

        x = x.transpose(1, 2).reshape(B, -1)
        x = self.dropout(x)  # Apply dropout before the fully connected layer

        x = self.fc(x)
        return x