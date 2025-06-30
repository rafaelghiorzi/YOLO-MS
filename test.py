from yolov8.tools.test import test

test(
    config_path="yolov8/config/coco_yolov8.yaml",
    checkpoint_path="C:/Users/rafae/Downloads/best.pt",
    output_dir="yolov8/output",
    source_path="yolov8/test/sample.png"
)