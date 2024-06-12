import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from helpers.setup import set_seeds


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expansion=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
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
    def __init__(self, device, num_blocks=3, out_features=39):
        super(HybridModel, self).__init__()
        # set device
        self.device = device

        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = 3
        out_channels = 32
        for i in range(num_blocks):
            self.conv_blocks.append(ConvBlock(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            in_channels = out_channels
            out_channels *= 2

        # Flatten to prepare for transformer
        self.flatten = nn.Flatten(start_dim=2)
        
        # Attention blocks
        self.attn1 = AttentionBlock(dim=in_channels, num_heads=4)
        self.attn2 = AttentionBlock(dim=in_channels, num_heads=4)

        # Classification head
        self.fc = nn.Linear(in_channels * 28 * 28, out_features)

    def forward(self, x):
        x = x.to(self.device)
        for conv in self.conv_blocks:
            x = conv(x)

        # Prepare for attention
        B, C, H, W = x.shape
        x = self.flatten(x)
        x = x.transpose(1, 2)  # (B, N, C)

        x = self.attn1(x)
        x = self.attn2(x)

        x = x.transpose(1, 2).reshape(B, -1)

        x = self.fc(x)
        return x


def create_effnetb0(device, out_features):
    # Get weights and model and send to target device
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    # Freeze base layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Seeds for reproducibility
    set_seeds(device=device)

    # Change classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1280, out_features=out_features)
    ).to(device)

    print(f"EffNetB0 model created successfully.")
    
    return model


def create_vit(device, out_features):
    # Get weights and model and send to target device
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    model = torchvision.models.vit_b_16(weights=weights).to(device)
    
    # Freeze base layers
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Seeds for reproducibility
    set_seeds(device=device)

    model.heads = nn.Linear(
        in_features=768, 
        out_features=out_features
    ).to(device)
    
    print(f"ViT model created successfully.")

    return model


def create_hybrid(device, out_features):

    # Seeds for reproducibility
    set_seeds(device=device)

    model = HybridModel(device, out_features)
    model.to(device)
    
    print(f"Hybrid model created successfully.")

    return model