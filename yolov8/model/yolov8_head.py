"""
YOLOv8 Detection Head Implementation

This module implements the YOLOv8 detection head that converts multi-scale feature maps
from the neck into final object detection predictions (bounding boxes + classifications).

Architecture Overview:
The head uses a decoupled design with separate branches for bounding box regression
and object classification, applied to three different feature map scales.

Key Components:
1. **Bounding Box Branch (self.box)**: Predicts object locations using DFL (Distribution Focal Loss)
2. **Classification Branch (self.cls)**: Predicts object class probabilities
3. **DFL Module**: Converts distribution predictions to precise bounding box coordinates

Input Features (from Neck):
- x[0]: [B, 256w, 80, 80]   - P3 features for small objects
- x[1]: [B, 512w, 40, 40]   - P4 features for medium objects  
- x[2]: [B, 512wr, 20, 20]  - P5 features for large objects

Head Architecture (per scale):
```
Feature Map → [Conv → Conv → Conv2d] → Box Predictions   (64 channels)
           → [Conv → Conv → Conv2d] → Class Predictions (80 channels)
```

Output Formats:

**Training Mode:**
- Returns list of 3 tensors: [B, 144, H, W] where 144 = 64 (box) + 80 (classes)
- Raw predictions for loss calculation

**Inference Mode:**
- Returns: [B, 8400, 84] where:
  - 8400 = total anchors across all scales (6400 + 1600 + 400)
  - 84 = 4 (box coords) + 80 (class probs)
  - Box format: [center_x, center_y, width, height]

Key Features:

**Decoupled Design:**
- Separate networks for box regression and classification
- Reduces task interference and improves accuracy

**Distribution Focal Loss (DFL):**
- Models bounding box coordinates as probability distributions
- Provides more precise localization than direct regression
- ch=16 means each coordinate uses 16-bin distribution

**Multi-Scale Detection:**
- Three detection heads for different object sizes
- Anchor-free design with grid-based predictions
- Automatic anchor generation based on feature map dimensions

**Anchor Generation:**
- Grid centers serve as anchor points
- Stride-aware coordinate scaling
- Offset=0.5 centers anchors in grid cells

Mathematical Flow:
1. Feature maps → Box/Class predictions
2. DFL converts box distributions to coordinates
3. Anchor generation creates grid coordinates
4. Final predictions: (anchors ± deltas) * strides

Where: w=width multiplier, r=ratio multiplier, B=batch size
"""
from yolov8.yolov8_utils import Conv, DFL ,yolo_params
import torch
from torch import nn

class Head(nn.Module):
    def __init__(self, version, ch=16, num_classes=80) -> None:
        super().__init__()
        self.ch = ch  # DFL channels
        self.coordinates = self.ch * 4  # number of coordinates (x, y, w, h)
        self.num_classes = num_classes  # 80 for COCO dataset
        self.no = self.coordinates + num_classes  # number of outputs per anchor
        self.stride = torch.zeros(3)
        depth, width, ratio = yolo_params(version)

        # for bounding boxes
        self.box=nn.ModuleList([
            nn.Sequential(Conv(int(256*width),self.coordinates,kernel_size=3,stride=1,padding=1),
                          Conv(self.coordinates,self.coordinates,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.coordinates,self.coordinates,kernel_size=1,stride=1)),

            nn.Sequential(Conv(int(512*width),self.coordinates,kernel_size=3,stride=1,padding=1),
                          Conv(self.coordinates,self.coordinates,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.coordinates,self.coordinates,kernel_size=1,stride=1)),

            nn.Sequential(Conv(int(512*width*ratio),self.coordinates,kernel_size=3,stride=1,padding=1),
                          Conv(self.coordinates,self.coordinates,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.coordinates,self.coordinates,kernel_size=1,stride=1))
        ])

        # for classification
        self.cls=nn.ModuleList([
            nn.Sequential(Conv(int(256*width),self.num_classes,kernel_size=3,stride=1,padding=1),
                          Conv(self.num_classes,self.num_classes,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.num_classes,self.num_classes,kernel_size=1,stride=1)),

            nn.Sequential(Conv(int(512*width),self.num_classes,kernel_size=3,stride=1,padding=1),
                          Conv(self.num_classes,self.num_classes,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.num_classes,self.num_classes,kernel_size=1,stride=1)),

            nn.Sequential(Conv(int(512*width*ratio),self.num_classes,kernel_size=3,stride=1,padding=1),
                          Conv(self.num_classes,self.num_classes,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.num_classes,self.num_classes,kernel_size=1,stride=1))
        ])

        # dfl
        self.dfl=DFL()

    def forward(self, x):
        # x = output of Neck = list of 3 tensors with different resolutions and channel dimensions
        # x[0] = [bs, ch0, w0, h0], x[1] = [bs, ch1, w1, h1], x[2] = [bs, ch2, w2, h2]

        for i in range(len(self.box)):              # Detection head i
            box = self.box[i](x[i])                 # [bs, num_coordinates, w, h]
            cls = self.cls[i](x[i])                 # [bs, num_classes, w, h]
            x[i] = torch.cat((box, cls), dim=1)     # [bs, num_coordinates + num_classes, w, h]

        if self.training:
            return x # During training, no DFL applied
        
        anchors, strides = (i.transpose(0, 1) for i in self.make_anchors(x, self.stride))

        # concatenate predictions from all detection layers
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2) #[bs, 4*self.ch + self.nc, sum_i(h[i]w[i])]
        
        # split out predictions for box and cls
        #           box=[bs,4×self.ch,sum_i(h[i]w[i])]
        #           cls=[bs,self.nc,sum_i(h[i]w[i])]
        box, cls = x.split(split_size=(4 * self.ch, self.num_classes), dim=1)


        a, b = self.dfl(box).chunk(2, 1)  # a=b=[bs,2×self.ch,sum_i(h[i]w[i])]
        a = anchors.unsqueeze(0) - a
        b = anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)
        
        return torch.cat(tensors=(box * strides, cls.sigmoid()), dim=1)

    def make_anchors(self, x, stride, offset=0.5):
        # Return a list of anchor centers and strides for each feature map
        assert x is not None
        anchor_tensor, stride_tensor = [], []
        dtype, device = x[0].dtype, x[0].device
        for i, stride in enumerate(stride):
            _, _, h, w = x[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + offset  # x coordinates of anchor centers
            sy = torch.arange(end=h, device=device, dtype=dtype) + offset  # y coordinates of anchor centers
            sy, sx = torch.meshgrid(sy, sx)                                # all anchor centers 
            anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_tensor), torch.cat(stride_tensor)