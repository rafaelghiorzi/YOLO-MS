"""
YOLOv8 Neck Implementation (Feature Pyramid Network + Path Aggregation Network)

This module implements the YOLOv8 neck that processes multi-scale features from the backbone
to create feature pyramids optimized for object detection at different scales.

Architecture Overview:
The neck combines FPN (top-down pathway) and PAN (bottom-up pathway) for bidirectional
feature fusion, enabling better detection of objects at multiple scales.

Flow Diagram:
                 Backbone Outputs
                P3(80×80)  P4(40×40)  P5(20×20)
                    ↓          ↓          ↓
    Top-Down:      P3 ←------- P4 ←------ P5
                    ↓          ↓          ↓
    Bottom-Up:     P3 ------→ P4 ------→ P5
                    ↓          ↓          ↓
                  out1       out2       out3

Input Features (from Backbone):
- P3 (x_res_1): [B, 64w, 80, 80]   - High resolution, low semantic
- P4 (x_res_2): [B, 128w, 40, 40]  - Medium resolution, medium semantic  
- P5 (x):       [B, 256wr, 20, 20] - Low resolution, high semantic

Output Features (to Detection Head):
- out1: [B, 256w, 80, 80]   - Enhanced P3 for small object detection
- out2: [B, 512w, 40, 40]   - Enhanced P4 for medium object detection  
- out3: [B, 512wr, 20, 20]  - Enhanced P5 for large object detection

Key Operations:
1. **Top-Down Pathway (FPN)**: 
   - Upsamples high-level features and fuses with lower-level features
   - Propagates strong semantic information to higher resolution layers

2. **Bottom-Up Pathway (PAN)**:
   - Downsamples enhanced features and fuses with higher-level features
   - Strengthens localization information in deeper layers

3. **Feature Fusion**:
   - Concatenation followed by C2f blocks for efficient feature integration
   - Maintains both semantic richness and spatial precision

Benefits:
- Multi-scale feature representation for detecting objects of various sizes
- Bidirectional information flow enhances both semantic and spatial features
- Efficient architecture balancing accuracy and computational cost

Where: w=width multiplier, r=ratio multiplier, B=batch size
"""
from yolov8.model.components import Conv, C2f, Upsample, yolo_params
import torch
from torch import nn
class Neck(nn.Module):
    def __init__(self, version):
        super().__init__()
        depth, width, ratio = yolo_params(version)
        self.up = Upsample()  # No trainable parameters
        self.c2f_1 = C2f(int(512 * width * (1+ ratio)), int(512 * width), num_bottlenecks=int(3 * depth), shortcut=False)
        self.c2f_2 = C2f(int(768 * width), int(256 * width), num_bottlenecks=int(3 * depth), shortcut=False)
        self.c2f_3 = C2f(int(768 * width), int(512 * width), num_bottlenecks=int(3 * depth), shortcut=False)
        self.c2f_4 = C2f(int(512 * width * (1+ ratio)), int(512 * width * ratio), num_bottlenecks=int(3 * depth), shortcut=False)

        self.conv1 = Conv(int(256 * width), int(256 * width), kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv(int(512 * width), int(512 * width), kernel_size=3, stride=2, padding=1)

    def forward(self, x_res_1, x_res_2, x):
        """
        Forward pass of the neck.

        x_res_1: Output from the first C2f block (P3) = out1
        x_res_2: Output from the second C2f block (P4) = out2
        x: Output from the SPPF block (P5) = out3
        """

        res_1 = x  # For residual connection
        x = self.up(x)

        x = torch.cat([x, x_res_2], dim=1)  # Concatenate P4 and upsampled P5

        res_2 = self.c2f_1(x)  # For residual connection
        x = self.up(res_2)
        x = torch.cat([x, x_res_1], dim=1)  # Concatenate P3 and upsampled P4
        
        out1 = self.c2f_2(x)

        x = self.conv1(out1)
        x = torch.cat([x, res_2], dim=1)
        out2 = self.c2f_3(x)
        x = self.conv2(out2)
        x = torch.cat([x, res_1], dim=1)
        out3 = self.c2f_4(x)

        return out1, out2, out3