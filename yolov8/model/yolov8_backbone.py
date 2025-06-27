"""
YOLOv8 Backbone Implementation

This module implements the YOLOv8 backbone network that extracts multi-scale features 
from input images for object detection.

Architecture Flow:
- Input: RGB image [B, 3, 640, 640]
- 5 Conv layers with stride=2 progressively downsample: 640→320→160→80→40→20
- 4 C2f layers refine features without changing spatial dimensions
- 1 SPPF layer captures multi-scale context in the deepest layer

Multi-Scale Outputs:
- P3 (out1): [B, 64w, 80, 80]   - 8x downsampled  - detects small objects
- P4 (out2): [B, 128w, 40, 40]  - 16x downsampled - detects medium objects  
- P5 (out3): [B, 256wr, 20, 20] - 32x downsampled - detects large objects

Model Variants (depth, width, ratio):
- Nano (n): 1/3, 1/4, 2.0    - Fastest, smallest
- Small (s): 1/3, 1/2, 2.0   - Good speed/accuracy balance
- Medium (m): 2/3, 3/4, 1.5  - Higher accuracy
- Large (l): 1.0, 1.0, 1.0   - High accuracy
- Extra (x): 1.0, 1.25, 1.0  - Highest accuracy

Where: w=width multiplier, r=ratio multiplier, B=batch size
"""
from yolov8.model.components import Conv, C2f, SPPF, yolo_params
from torch import nn
  
class Backbone(nn.Module):
    """
    Backbone of the YOLOv8 model.
    """
    def __init__(self, version, in_channels=3, shortcut=True) -> None:
        super().__init__()
        depth,width,ratio = yolo_params(version)

        # Conv layers
        self.conv0 = Conv(in_channels, int(64 * width), kernel_size=3, stride=2, padding=1)
        self.conv1 = Conv(int(64 * width), int(128 * width), kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv(int(128 * width), int(256 * width), kernel_size=3, stride=2, padding=1)
        self.conv5 = Conv(int(256 * width), int(512 * width), kernel_size=3, stride=2, padding=1)
        self.conv7 = Conv(int(512 * width), int(512 * width * ratio), kernel_size=3, stride=2, padding=1)

        # C2f layers
        self.c2f_2 = C2f(int(128 * width), int(128 * width), num_bottlenecks=int(3 * depth), shortcut=True)
        self.c2f_4 = C2f(int(256 * width), int(256 * width), num_bottlenecks=int(6 * depth), shortcut=True)
        self.c2f_6 = C2f(int(512 * width), int(512 * width), num_bottlenecks=int(6 * depth), shortcut=True)
        self.c2f_8 = C2f(int(512 * width * ratio), int(512 * width * ratio), num_bottlenecks=int(3 * depth), shortcut=True)

        # SPPF layer
        self.sppf = SPPF(int(512 * width * ratio), int(512 * width * ratio), kernel_size=5)

    def forward(self, x):
        """
        Forward pass of the backbone.
        """
        x = self.conv0(x)
        x = self.conv1(x)

        x = self.c2f_2(x)
        x = self.conv3(x)

        out1 = self.c2f_4(x) # Keep for the output

        x = self.conv5(out1)

        out2 = self.c2f_6(x) # Keep for the output

        x = self.conv7(out2)
        x = self.c2f_8(x)
        out3 = self.sppf(x)

        return out1, out2, out3