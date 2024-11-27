import torch.nn as nn
import torch.nn.functional as F

class ImprovedSegmentationModel(nn.Module):
    """
    Improved segmentation model using deeper convolutional layers,
    skip connections, and upsampling.
    """
    def __init__(self, num_classes):
        super(ImprovedSegmentationModel, self).__init__()
        
        # Encoder (Downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder (Upsampling)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Final output
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # [Batch, 64, H, W]
        pool1 = self.pool1(enc1)  # [Batch, 64, H/2, W/2]

        enc2 = self.enc2(pool1)  # [Batch, 128, H/2, W/2]
        pool2 = self.pool2(enc2)  # [Batch, 128, H/4, W/4]

        # Bottleneck
        bottleneck = self.bottleneck(pool2)  # [Batch, 256, H/4, W/4]

        # Decoder
        up2 = self.up2(bottleneck)  # [Batch, 128, H/2, W/2]
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))  # Skip connection

        up1 = self.up1(dec2)  # [Batch, 64, H, W]
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))  # Skip connection

        # Final output
        final = self.final(dec1)  # [Batch, num_classes, H, W]
        return final
