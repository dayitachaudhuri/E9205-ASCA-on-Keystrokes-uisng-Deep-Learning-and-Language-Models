import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
# Assume AudioKeystrokeDataset is defined in AudioKeystrokeDataset.py
from AudioKeystrokeDataset.AudioKeystrokeDataset import AudioKeystrokeDataset

##############################################
# Define Convolutional (MBConv) and Transformer Blocks
##############################################

class MBConv(nn.Module):
    """
    A Mobile Inverted Bottleneck Convolution (MBConv) block.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=4):
        super(MBConv, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.conv = nn.Sequential(
            # Expansion phase
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # Projection phase
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            return out + x
        return out

class TransformerBlock(nn.Module):
    """
    A transformer block that first normalizes and then applies multi-head attention 
    followed by a feed-forward network.
    """
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

##############################################
# Define the Modified CoAtNet Architecture
##############################################

class CoAtNet(nn.Module):
    def __init__(self, num_classes=36, in_channels=1):
        super(CoAtNet, self).__init__()
        # Stem: initial convolution to reduce spatial dimensions
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Stage 1: Convolutional blocks (MBConv)
        self.stage1 = nn.Sequential(
            MBConv(32, 64, stride=2),
            MBConv(64, 64, stride=1)
        )
        # Stage 2: Transformer blocks
        self.stage2 = nn.Sequential(
            TransformerBlock(embed_dim=64, num_heads=4),
            TransformerBlock(embed_dim=64, num_heads=4)
        )
        # Stage 3: Further convolutional blocks (MBConv)
        self.stage3 = nn.Sequential(
            MBConv(64, 128, stride=2),
            MBConv(128, 128, stride=1)
        )
        # Classification head: global pooling and a linear classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x expected shape: (B, 1, H, W)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

##############################################
# Define a Trainer Class for Model Training
##############################################

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for data, targets in self.train_loader:
            # Move data to device and ensure it has a channel dimension
            data, targets = data.to(self.device), targets.to(self.device)
            if len(data.shape) == 3:  # add channel dimension if missing
                data = data.unsqueeze(1)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                if len(data.shape) == 3:
                    data = data.unsqueeze(1)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def train(self, num_epochs, save_path=None):
        best_val_acc = 0.0
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            if save_path and val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model with Val Acc: {best_val_acc:.4f}")

##############################################
# Main Function to Setup Data, Model, and Training
##############################################

def main():
    # For reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Hyperparameters
    num_epochs = 20
    batch_size = 32
    learning_rate = 1e-3
    
    # Initialize your dataset (adjust 'dataset_path' as needed)
    dataset = AudioKeystrokeDataset(dataset_path='data/Mac/Raw Data', energy_threshold=100)
    
    # Split dataset into training and validation sets (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the modified CoAtNet model
    # The number of classes is set from the dataset label mapping
    model = CoAtNet(num_classes=len(dataset.label2idx), in_channels=1)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create Trainer and start training
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
    trainer.train(num_epochs, save_path='best_coatnet_model.pth')

if __name__ == '__main__':
    main()
