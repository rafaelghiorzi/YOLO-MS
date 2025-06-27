from yolov8.test.test_model import analyze_complete_model
from yolov8.yolov8 import YOLOv8
import torch

def visualize(version):

    print(f"\n{'='*60}")
    print(f"Testing YOLOv8-{version.upper()}")
    print(f"{'='*60}")
    
    try:
        outputs = analyze_complete_model("yolov8/test/sample.png", version=version)
        print(f"\n✅ YOLOv8-{version.upper()} analysis complete!")
    except FileNotFoundError:
        print("❌ Image file not found. Please ensure 'sample.png' exists in the yolov8 folder.")
        print("💡 Tip: You can use any .jpg, .png, or .jpeg image file.")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

def outputs(model_version, num_classes_coco, dfl_channels):
    print(f"\n{'='*60}")
    print(f"Initializing YOLOv8-{model_version} with {num_classes_coco} classes...")
    
    try:
        model = YOLOv8(version=model_version, num_classes=num_classes_coco, dfl_ch=dfl_channels)
        
        print(f"YOLOv8-{model_version} Model initialized successfully.")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"Total parameters: {total_params:.2f} million")

        # Test input
        dummy_input = torch.randn(1, 3, 640, 640)
        print(f"\nTesting forward pass with input shape: {dummy_input.shape}")
        
        # Training mode
        model.train()
        with torch.no_grad():
            predictions_train = model(dummy_input)
        
        print("\n📊 Training Mode Outputs:")
        total_anchors = 0
        for i, p in enumerate(predictions_train):
            anchors = p.shape[2] * p.shape[3]  # H * W
            total_anchors += anchors
            downsampling = 2**(i+3)  # 8x, 16x, 32x
            print(f"  P{i+3}: {p.shape} - {anchors:,} anchors - {downsampling}x downsampled")
        print(f"  Total training anchors: {total_anchors:,}")
        
        # Evaluation mode  
        model.eval()
        # Set proper strides (should match your backbone output strides)
        model.head.stride = torch.tensor([8., 16., 32.])
        
        with torch.no_grad():
            predictions_eval = model(dummy_input)
        
        print(f"\n📊 Evaluation Mode Output:")
        print(f"  Shape: {predictions_eval.shape}")
        print(f"  Format: [batch_size, total_anchors, features]")
        print(f"  - Batch size: {predictions_eval.shape[0]}")
        print(f"  - Total anchors: {predictions_eval.shape[1]:,}")
        print(f"  - Features: {predictions_eval.shape[2]} (4 box coords + {num_classes_coco} classes)")
        
        # Verify anchor count matches
        expected_anchors = sum([(640//(2**(i+3)))**2 for i in range(3)])  # 80²+40²+20² = 6400+1600+400 = 8400
        print(f"  - Expected anchors: {expected_anchors:,} ✅" if predictions_eval.shape[1] == expected_anchors 
              else f"  - Expected anchors: {expected_anchors:,} ❌")
        
        print(f"\n✅ YOLOv8-{model_version} forward pass test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Visualize model outputs
    visualize('n')  # Change to 's', 'm', 'l', 'x' for other versions
    
    # Test YOLOv8 with different configurations
    versions = ['n', 's', 'm', 'l', 'x']  # Small to Extra Large
    num_classes_coco = 80  # COCO dataset classes
    dfl_channels = 16  # DFL channels
    
    for version in versions:
        outputs(version, num_classes_coco, dfl_channels)