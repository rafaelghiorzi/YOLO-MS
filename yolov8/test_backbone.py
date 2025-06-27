"""
Visualize YOLOv8 Backbone Feature Maps

This script loads a real image, passes it through the YOLOv8 backbone,
and visualizes the feature maps at different scales.
"""
from typing import cast
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model.yolov8_backbone import Backbone

def load_and_preprocess_image(image_path, size=640):
    """Load and preprocess image for YOLOv8 backbone"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        # Optional: normalize like YOLO training
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms and add batch dimension
    tensor = cast(torch.Tensor, transform(image))
    tensor = tensor.unsqueeze(0)
    return tensor, image

def visualize_feature_maps(feature_maps, original_image, num_channels=8):
    """Visualize feature maps from backbone outputs"""
    
    fig, axes = plt.subplots(4, num_channels, figsize=(20, 10))
    fig.suptitle('YOLOv8 Backbone Feature Maps', fontsize=16)
    
    # Show original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Clear unused slots in first row
    for i in range(1, num_channels):
        axes[0, i].axis('off')
    
    # Visualize each output scale
    scale_names = ['P3 (Small Objects)', 'P4 (Medium Objects)', 'P5 (Large Objects)']
    
    for scale_idx, (feature_map, scale_name) in enumerate(zip(feature_maps, scale_names)):
        row = scale_idx + 1
        
        # Convert to numpy and remove batch dimension
        feature_np = feature_map.detach().cpu().numpy()[0]  # Shape: [C, H, W]
        
        # Show first num_channels feature maps
        for ch in range(min(num_channels, feature_np.shape[0])):
            axes[row, ch].imshow(feature_np[ch], cmap='viridis')
            axes[row, ch].set_title(f'{scale_name}\nCh {ch+1}/{feature_np.shape[0]}')
            axes[row, ch].axis('off')
        
        # Clear unused channels
        for ch in range(feature_np.shape[0], num_channels):
            axes[row, ch].axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_backbone_output(image_path):
    """Complete analysis of backbone processing"""
    print("🖼️  Loading and preprocessing image...")
    
    # Load image
    input_tensor, original_image = load_and_preprocess_image(image_path)
    print(f"Input shape: {input_tensor.shape}")
    
    # Create backbone
    print("\n🧠 Creating YOLOv8-Nano backbone...")
    backbone = Backbone(version='n')
    backbone.eval()  # Set to evaluation mode
    
    # Forward pass
    print("\n⚡ Processing through backbone...")
    with torch.no_grad():  # Disable gradients for inference
        out1, out2, out3 = backbone(input_tensor)
    
    # Print output information
    print(f"\n📊 Backbone Outputs:")
    print(f"P3 (Small objects):  {out1.shape} - {out1.shape[2]/640:.1f}x downsampled")
    print(f"P4 (Medium objects): {out2.shape} - {out2.shape[2]/640:.1f}x downsampled")
    print(f"P5 (Large objects):  {out3.shape} - {out3.shape[2]/640:.1f}x downsampled")
    
    # Calculate receptive fields (approximate)
    print(f"\n🔍 Receptive Field Analysis:")
    print(f"P3: ~{640/out1.shape[2]*8:.0f}px receptive field - detects small objects")
    print(f"P4: ~{640/out2.shape[2]*16:.0f}px receptive field - detects medium objects")
    print(f"P5: ~{640/out3.shape[2]*32:.0f}px receptive field - detects large objects")
    
    # Visualize
    print(f"\n🎨 Visualizing feature maps...")
    visualize_feature_maps([out1, out2, out3], original_image)
    
    return out1, out2, out3

# Example usage
if __name__ == "__main__":
    # You can use any image path here
    # For testing, you could download a sample image or use a webcam capture
    
    # Option 1: Use a sample image (replace with your image path)
    image_path = "yolov8/sample.png"  # Replace with your image path
    
    # Analyze the backbone
    try:
        outputs = analyze_backbone_output("yolov8/sample.png")
        print("\n✅ Analysis complete!")
    except FileNotFoundError:
        print("❌ Image file not found. Please provide a valid image path.")
        print("💡 Tip: You can use any .jpg, .png, or .jpeg image file.")