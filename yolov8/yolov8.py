from model.yolov8_backbone import Backbone
from model.yolov8_neck import Neck
from model.yolov8_head import Head
from torch import nn

class YOLOv8(nn.Module):
    def __init__(self, version) -> None:
        super().__init__()
        self.backbone = Backbone(version)
        self.neck = Neck(version)
        self.head = Head(version)

    def forward(self, x):
        x = self.backbone(x)  # Output from Backbone
        x = self.neck(x[0], x[1], x[2])  # Output from Neck
        return self.head(list(x))
    
model = YOLOv8(version='x')  # Example for YOLOv8-Extra
print(f"YOLOv8 Model: {sum(p.numel() for p in model.parameters()) / 1e6} million parameters")