# model.py
import torch.nn as nn

class SimpleSegmentationModel(nn.Module):
    """
    Simple segmentation model using convolutional layers.
    """
    def __init__(self, num_classes):
        super(SimpleSegmentationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x
