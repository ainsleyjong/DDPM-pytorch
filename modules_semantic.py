import torch
import torch.nn as nn
import torch.nn.functional as F

class SPADE(nn.Module):
    """
    A basic SPADE (Spatially-Adaptive Denormalization) layer.
    It first normalizes the input feature map using GroupNorm
    (or another normalization with affine disabled) and then
    modulates it using parameters (gamma and beta) computed from the semantic mask.
    """
    def __init__(self, norm_nc, label_nc):
        """
        norm_nc: Number of channels in the feature map to be normalized.
        label_nc: Number of channels in the semantic label mask (e.g., number of classes if one-hot encoded).
        """
        super().__init__()
        # Parameter-free normalization (using GroupNorm here)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=norm_nc, affine=False)
        
        # Shared MLP to process the segmentation (semantic mask)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Separate conv layers to produce gamma and beta parameters
        self.mlp_gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta  = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        
    def forward(self, x, segmap):
        # Normalize input feature map (x)
        normalized = self.norm(x)
        # Resize the semantic mask to match the spatial dimensions of x if needed
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta  = self.mlp_beta(actv)
        # Apply the spatially-adaptive scale and bias
        out = normalized * (1 + gamma) + beta
        return out

class UNetSemantic(nn.Module): #might ignore
    def __init__(self, in_channels=3, out_channels=3, label_nc=35):
        """
        in_channels: Number of image channels.
        out_channels: Number of output channels.
        label_nc: Number of channels for semantic mask (for one-hot encoding, equals number of classes).
        """
        super().__init__()
        # Encoder: process the noisy image without semantic mask
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
        # Decoder: inject semantic information using SPADE
        self.dec1 = SPADE(256, label_nc)  # Example: apply SPADE on bottleneck features
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = SPADE(128, label_nc)  # Example: re-normalize features after upsampling
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # In a more complex implementation you’d likely have more layers
        # and residual blocks with SPADE injection in the decoder.
        
        # For this simple example, we assume a very basic U-Net:
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x, segmap):
        # Encoder stage
        x1 = self.enc1(x)       # [B, 64, H, W]
        x2 = self.enc2(x1)      # [B, 128, H/2, W/2]
        x3 = self.bottleneck(x2)  # [B, 256, H/2, W/2]
        
        # Decoder stage: first apply a convolution to prepare for SPADE conditioning
        dec_features = self.decoder_conv(x3)  # [B, 64, H/2, W/2]
        # Upsample features to match input resolution
        up_features = self.up1(dec_features)  # [B, 64, H, W]
        # Apply SPADE conditioning using the semantic mask
        conditioned = self.dec2(up_features, segmap)
        out = self.final_conv(conditioned)
        return out

class SpadeResBlock(nn.Module):
    """
    A residual block that uses SPADE normalization in its processing.
    It takes an input feature map and a semantic mask, applies SPADE normalization
    before each convolution, and then adds a learned shortcut.
    """
    def __init__(self, fin, fout, label_nc):
        """
        fin: Number of input channels.
        fout: Number of output channels.
        label_nc: Number of channels for the semantic mask.
        """
        super().__init__()
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)
        
        # Two convolutional layers for the main branch
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        
        # SPADE layers to condition the normalization
        self.spade_0 = SPADE(fin, label_nc)
        self.spade_1 = SPADE(fmiddle, label_nc)
        
        # If a learned shortcut is needed, define it
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
            self.spade_s = SPADE(fin, label_nc)
    
    def shortcut(self, x, segmap):
        if self.learned_shortcut:
            x_s = self.spade_s(x, segmap)
            x_s = self.conv_s(x_s)
            return x_s
        else:
            return x
    
    def forward(self, x, segmap):
        # Shortcut path
        x_s = self.shortcut(x, segmap)
        
        # Main branch with SPADE conditioning before convolutions
        dx = self.spade_0(x, segmap)
        dx = F.relu(dx)
        dx = self.conv_0(dx)
        dx = self.spade_1(dx, segmap)
        dx = F.relu(dx)
        dx = self.conv_1(dx)
        
        out = x_s + dx
        return out

