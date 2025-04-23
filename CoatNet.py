import torch
from torch import nn

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=4):
        super(MBConv, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.conv = nn.Sequential(
            # Expansion phase
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # Projection phase
            nn.Dropout(0.1),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
 
    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            return out + x
        return out
 
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x):
        # x is expected to have shape (B, C, H, W)
        B, C, H, W = x.shape
        # Flatten spatial dimensions: (S, B, C) where S = H*W
        x_flat = x.flatten(2).permute(2, 0, 1)
        # Apply multi-head self-attention with residual connection
        attn_out, _ = self.attn(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))
        x_flat = x_flat + self.dropout(attn_out)
        # Feed-forward network with residual connection
        ff_out = self.ff(self.norm2(x_flat))
        x_flat = x_flat + self.dropout(ff_out)
        # Reshape back to (B, C, H, W)
        x = x_flat.permute(1, 2, 0).view(B, C, H, W)
        return x
 
class CoAtNet(nn.Module):
    def __init__(self, num_classes=36, num_devices=8, in_channels=1, device_embed_dim=8):
        super(CoAtNet, self).__init__()
        self.device_embedding = nn.Embedding(num_devices, device_embed_dim)

        # Stem and other blocks remain the same
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(
            MBConv(32, 64, stride=2),
            MBConv(64, 64, stride=1)
        )
        self.stage2 = nn.Sequential(
            TransformerBlock(embed_dim=64, num_heads=4),
            TransformerBlock(embed_dim=64, num_heads=4)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 + device_embed_dim, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, device_id):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        device_embed = self.device_embedding(device_id)
        x = torch.cat((x, device_embed), dim=1)

        return self.fc(x)