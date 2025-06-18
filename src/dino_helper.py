import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


# Define the segmentation model with DINOv2 backbone
class DINOv2Segmentation(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initialize DINOv2 backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.feature_dim = 384  # DINOv2-small feature dimension
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Decoder network with skip connections
        self.decoder = nn.Sequential(
            # First block
            nn.Conv2d(self.feature_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Second block
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Third block
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Final 1x1 conv for classification
            nn.Conv2d(64, 7, kernel_size=1)
        )
        
    def forward(self, x):
        # Get features from DINOv2 backbone
        features_dict = self.backbone.forward_features(x)
        
        # Extract patch tokens from the dictionary
        features = features_dict['x_norm_patchtokens']  # Shape: [B, N, D]
        B = x.shape[0]
        
        # Reshape features from [B, N, D] to [B, D, H, W]
        # N = H * W = 16 * 16 = 256 for 224x224 input
        features = features.permute(0, 2, 1)  # [B, D, N]
        features = features.reshape(B, -1, 16, 16)  # [B, D, H, W]
        
        # Apply decoder
        x = self.decoder(features)
        
        # Upsample to input size
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        return x