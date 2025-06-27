from yolov8.model.yolov8_backbone import Backbone
from yolov8.model.yolov8_neck import Neck
from yolov8.model.yolov8_head import Head
from torch import nn


class YOLOv8(nn.Module):
    def __init__(self, version: str, num_classes: int, dfl_ch: int = 16) -> None:
        """
        Initializes the YOLOv8 model.

        Args:
            version (str): Model version/size (e.g., 's', 'm', 'l', 'x').
                           Used to determine depth and width multipliers.
            num_classes (int): Number of classes for the detection head.
            dfl_ch (int): Number of channels for DFL module in the head. Default is 16.
        """
        super().__init__()
        self.backbone = Backbone(version)
        self.neck = Neck(version)
        self.head = Head(version=version, num_classes=num_classes, ch=dfl_ch) 

    def forward(self, x):
        features = self.backbone(x)  
        if not isinstance(features, (list, tuple)) or len(features) < 3:
            raise ValueError(f"Backbone output must be a list/tuple of at least 3 feature maps. Got: {type(features)}")

        neck_output = self.neck(features[0], features[1], features[2])  # Output from Neck
        if not isinstance(neck_output, (list, tuple)):
            neck_output = [neck_output] # Wrap if it's a single tensor, though Neck usually outputs multiple

        return self.head(list(neck_output))