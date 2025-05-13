import torch
import torch.nn as nn
from modules_semantic import SpadeResBlock

class UNetSemantic(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, label_nc=35):
        """
        in_channels: Number of channels in the input image.
        out_channels: Number of channels in the generated image.
        label_nc: Number of channels for the semantic mask (e.g., one-hot encoded classes).
        """
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.dec1 = SpadeResBlock(256, 128, label_nc)
        self.dec2 = SpadeResBlock(128, 64, label_nc)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x, segmap):
        # Encode image only (ignore segmap here)
        x1 = self.enc1(x) 
        x2 = self.enc2(x1) 
        x_bottleneck = self.bottleneck(x2)
        
        d1 = self.dec1(x_bottleneck, segmap) 
        d1_up = nn.functional.interpolate(d1, scale_factor=2, mode='nearest')
        d2 = self.dec2(d1_up, segmap)
        out = self.out_conv(d2)
        return out