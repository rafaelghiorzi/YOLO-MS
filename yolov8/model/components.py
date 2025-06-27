"""
YOLOv8 Building Blocks and Components

This module contains the fundamental building blocks used throughout the YOLOv8 architecture.
These components are designed to be reusable across different parts of the model (backbone, neck, head).

Components:
-----------
Conv: Basic convolution block with BatchNorm and SiLU activation
    - Standard building block for feature extraction
    - Formula: Conv2d → BatchNorm2d → SiLU
    
Bottleneck: Residual block with optional skip connection
    - Two 3x3 convolutions with residual connection
    - Enables deeper networks without gradient degradation
    
C2f: Cross-Stage Partial bottleneck with 2 convolutions
    - Efficient feature processing with multiple paths
    - Splits channels, processes through bottlenecks, then concatenates
    - Balances accuracy and computational efficiency
    
SPPF: Spatial Pyramid Pooling - Fast
    - Captures multi-scale spatial information efficiently
    - Uses sequential max pooling instead of parallel (faster than SPP)
    - Aggregates features at different receptive field sizes

Upsample: Nearest neighbor upsampling
    - Upscales feature maps by a specified scale factor
    - Used for feature map resizing in the neck and head

yolo_params: Function to get YOLOv8 model parameters based on version
    - Returns depth, width, and ratio multipliers for different model sizes
    - Supports 'n' (Nano), 's' (Small), 'm' (Medium), 'l' (Large), 'x' (Extra)
    - Helps in configuring the model architecture dynamically

DFL: Distribution Focal Loss
    - Considers the predicted bbox coordinates as a probability distribution
    - At inference, it samples from the distribution to get refined coordinates.

Key Features:
- All components preserve or enhance feature representation
- Designed for efficient inference and training
- Compatible with different input sizes and channel counts
- Optimized for object detection tasks

Usage:
    from yolov8_utils import Conv, C2f, SPPF, Bottleneck
    
    # Basic convolution
    conv = Conv(3, 64, kernel_size=3, stride=2)
    
    # Feature refinement
    c2f = C2f(64, 128, num_bottlenecks=3)
    
    # Multi-scale pooling
    sppf = SPPF(512, 512, kernel_size=5)

    # Upsampling feature maps
    upsample = Upsample(scale_factor=2, mode='nearest')

    # Yolo Parameters
    depth, width, ratio = yolo_params('n')  # Get parameters for YOLOv8-Nano
"""

import torch
from torch import nn

# Conv: Conv2d + BatchNorm2d + SiLU activation
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, activation=True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.activation = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

# Bottleneck: stack of 2 convolutions with a skip connection
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True) -> None:
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = shortcut

    def forward(self, x):
        x_in = x  # for skip connection
        x= self.conv1(x)
        x = self.conv2(x)
        if self.shortcut:
            x += x_in
        return x

# C2f: Cross-stage partial bottleneck with 2 convolutions
class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_bottlenecks, shortcut=True) -> None:
        super().__init__()
        self.mid_channels = out_channels // 2
        self.num_bottlenecks = num_bottlenecks

        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # Sequence of bottlenecks layers
        self.m = nn.ModuleList([Bottleneck(self.mid_channels, self.mid_channels) for _ in range(num_bottlenecks)])

        self.conv2 = Conv((num_bottlenecks + 2) * out_channels // 2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        # Split x along the channel dimension
        x1, x2 = x[:, :x.shape[1] // 2, :, :], x[:, x.shape[1] // 2:, :, :]

        # List of outputs
        outputs = [x1, x2] # x1 is fed to the bottlenecks
        for i in range(self.num_bottlenecks):
            x1 = self.m[i](x1)
            outputs.insert(0, x1)
        outputs = torch.cat(outputs, dim=1)  # Concatenate along the channel dimension

        out = self.conv2(outputs)
        return out

# SPPF: Spatial Pyramid Pooling - Fast
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5) -> None:
        # Kernel size is the size of the pooling window
        super().__init__()

        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        # Concatenate the pooled values and apply to conv2
        self.conv2 = Conv(hidden_channels * 4, out_channels, kernel_size=1, stride=1, padding=0)

        # MaxPool with 3 different scales, 3 different kernel sizes
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2, dilation=1, ceil_mode=False)

    def forward(self, x):
        x = self.conv1(x)

        # Apply max pooling with different kernel sizes
        x1 = self.m(x)
        x2 = self.m(x1)
        x3 = self.m(x2)
        # Concatenate the pooled values
        y = torch.cat([x, x1, x2, x3], dim=1)

        # Final Conv layer
        y = self.conv2(y)
        return y

# Upsample: nearest neighbor interpolation with scale factor
class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest') -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

class DFL(nn.Module):
    def __init__(self, ch=16) -> None:
        super().__init__()
        self.ch = ch

        self.conv = nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)

        # Initialize conv with [0, ..., ch-1]
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        """
        Forward pass of the Distribution Focal Loss (DFL).
        x: Input tensor with shape [B, ch * 4, A]
        - B: Batch size
        - ch: Number of channels (e.g., 16)
        - A: Number of anchors (e.g., 80x80 grid)
        Returns:
        - Output tensor with shape [B, 4, A]
        """

        # x must have num_channels = ch * 4
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(1, 2)

        # Take softmax on channel dimension
        x = x.softmax(dim=1)
        x = self.conv(x)
        return x.view(b, 4, a)

def yolo_params(version):
    """
    Returns the parameters for the YOLOv8 model based on the version.
    return depth,width,ratio based on the version
    """
    if version == 'n':
        return 1/3, 1/4, 2.0
    elif version == 's':
        return 1/3, 1/2, 2.0
    elif version == 'm':
        return 2/3, 3/4, 1.5
    elif version == 'l':
        return 1.0, 1.0, 1.0
    elif version == 'x':
        return 1.0, 1.25, 1.0
    else:
        raise ValueError(f"Unknown YOLOv8 version: {version}")