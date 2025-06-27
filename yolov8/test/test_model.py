"""
Visualize Complete YOLOv8 Model Pipeline

This script loads a real image, passes it through the complete YOLOv8 model
(backbone + neck + head), and visualizes the outputs at each stage.
"""
from typing import cast
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from yolov8.model.yolov8_backbone import Backbone
from yolov8.model.yolov8_neck import Neck
from yolov8.model.yolov8_head import Head
from yolov8.yolov8 import YOLOv8

def load_and_preprocess_image(image_path, size=640):
    """Load and preprocess image for YOLOv8"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    
    # Apply transforms and add batch dimension
    tensor = cast(torch.Tensor, transform(image))
    tensor = tensor.unsqueeze(0)
    return tensor, image

def visualize_backbone_features(backbone_outputs, original_image, num_channels=6):
    """Visualize backbone feature maps"""
    
    fig, axes = plt.subplots(4, num_channels, figsize=(18, 12))
    fig.suptitle('YOLOv8 Backbone Feature Maps', fontsize=16)
    
    # Show original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Clear unused slots in first row
    for i in range(1, num_channels):
        axes[0, i].axis('off')
    
    # Visualize each backbone output scale
    scale_names = ['P3 - Backbone (Small)', 'P4 - Backbone (Medium)', 'P5 - Backbone (Large)']
    
    for scale_idx, (feature_map, scale_name) in enumerate(zip(backbone_outputs, scale_names)):
        row = scale_idx + 1
        
        # Convert to numpy and remove batch dimension
        feature_np = feature_map.detach().cpu().numpy()[0]  # Shape: [C, H, W]
        
        # Show first num_channels feature maps
        for ch in range(min(num_channels, feature_np.shape[0])):
            axes[row, ch].imshow(feature_np[ch], cmap='viridis')
            axes[row, ch].set_title(f'{scale_name}\nCh {ch+1}/{feature_np.shape[0]} | {feature_np.shape[1]}x{feature_np.shape[2]}')
            axes[row, ch].axis('off')
        
        # Clear unused channels
        for ch in range(min(num_channels, feature_np.shape[0]), num_channels):
            axes[row, ch].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_neck_features(neck_outputs, num_channels=6):
    """Visualize neck feature maps"""
    
    fig, axes = plt.subplots(3, num_channels, figsize=(18, 9))
    fig.suptitle('YOLOv8 Neck (FPN + PAN) Feature Maps', fontsize=16)
    
    # Visualize each neck output scale
    scale_names = ['P3 - Neck (Small)', 'P4 - Neck (Medium)', 'P5 - Neck (Large)']
    
    for scale_idx, (feature_map, scale_name) in enumerate(zip(neck_outputs, scale_names)):
        row = scale_idx
        
        # Convert to numpy and remove batch dimension
        feature_np = feature_map.detach().cpu().numpy()[0]  # Shape: [C, H, W]
        
        # Show first num_channels feature maps
        for ch in range(min(num_channels, feature_np.shape[0])):
            axes[row, ch].imshow(feature_np[ch], cmap='plasma')
            axes[row, ch].set_title(f'{scale_name}\nCh {ch+1}/{feature_np.shape[0]} | {feature_np.shape[1]}x{feature_np.shape[2]}')
            axes[row, ch].axis('off')
        
        # Clear unused channels
        for ch in range(min(num_channels, feature_np.shape[0]), num_channels):
            axes[row, ch].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_head_outputs(head_output, training_outputs=None):
    """Visualize head outputs"""
    
    if training_outputs is not None:
        # Training mode visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('YOLOv8 Head Outputs (Training Mode)', fontsize=16)
        
        scale_names = ['P3 Head Output', 'P4 Head Output', 'P5 Head Output']
        
        for i, (output, name) in enumerate(zip(training_outputs, scale_names)):
            # Box predictions (first 64 channels)
            box_features = output[0, :64, :, :].detach().cpu().numpy()
            box_avg = np.mean(box_features, axis=0)
            
            # Class predictions (last 80 channels) 
            cls_features = output[0, 64:, :, :].detach().cpu().numpy()
            cls_avg = np.mean(cls_features, axis=0)
            
            axes[0, i].imshow(box_avg, cmap='coolwarm')
            axes[0, i].set_title(f'{name}\nBox Predictions Avg')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(cls_avg, cmap='coolwarm')
            axes[1, i].set_title(f'{name}\nClass Predictions Avg')
            axes[1, i].axis('off')
    
    else:
        # Inference mode visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('YOLOv8 Head Output (Inference Mode)', fontsize=16)
        
        # head_output shape: [B, 8400, 84]
        output_np = head_output.detach().cpu().numpy()[0]  # Remove batch dimension: [8400, 84]
        
        # Box coordinates (first 4 columns)
        box_coords = output_np[:, :4]  # [8400, 4] - [x, y, w, h]
        
        # Class probabilities (last 80 columns)
        class_probs = output_np[:, 4:]  # [8400, 80]
        
        # Visualize box coordinates distribution
        axes[0, 0].hist(box_coords[:, 0], bins=50, alpha=0.7, label='X coords')
        axes[0, 0].set_title('X Coordinate Distribution')
        axes[0, 0].set_xlabel('X coordinate')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].hist(box_coords[:, 1], bins=50, alpha=0.7, label='Y coords', color='orange')
        axes[0, 1].set_title('Y Coordinate Distribution')
        axes[0, 1].set_xlabel('Y coordinate')
        axes[0, 1].set_ylabel('Frequency')
        
        axes[1, 0].hist(box_coords[:, 2], bins=50, alpha=0.7, label='Width', color='green')
        axes[1, 0].set_title('Width Distribution')
        axes[1, 0].set_xlabel('Width')
        axes[1, 0].set_ylabel('Frequency')
        
        axes[1, 1].hist(box_coords[:, 3], bins=50, alpha=0.7, label='Height', color='red')
        axes[1, 1].set_title('Height Distribution')
        axes[1, 1].set_xlabel('Height')
        axes[1, 1].set_ylabel('Frequency')
        
        # Show class probabilities heatmap
        fig2, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Show max class probability for each anchor
        max_class_probs = np.max(class_probs, axis=1)
        ax.hist(max_class_probs, bins=50, alpha=0.7)
        ax.set_title('Maximum Class Probability Distribution\n(Confidence scores across all 8400 anchors)')
        ax.set_xlabel('Max Class Probability')
        ax.set_ylabel('Number of Anchors')
        
        # Add statistics
        mean_conf = np.mean(max_class_probs)
        max_conf = np.max(max_class_probs)
        ax.axvline(mean_conf, color='red', linestyle='--', label=f'Mean: {mean_conf:.3f}')
        ax.axvline(max_conf, color='green', linestyle='--', label=f'Max: {max_conf:.3f}')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    plt.tight_layout()
    plt.show()

def analyze_complete_model(image_path, version='n', test_training_mode=True):
    """Complete analysis of YOLOv8 model pipeline"""
    print("🖼️  Loading and preprocessing image...")
    
    # Load image
    input_tensor, original_image = load_and_preprocess_image(image_path)
    print(f"Input shape: {input_tensor.shape}")
    
    # Create individual components for detailed analysis
    print(f"\n🧠 Creating YOLOv8-{version.upper()} components...")
    backbone = Backbone(version=version)
    neck = Neck(version=version)
    head = Head(version=version)
    
    # Also create complete model
    complete_model = YOLOv8(version=version, num_classes=80, dfl_ch=16)
    
    print(f"📊 Model Statistics:")
    total_params = sum(p.numel() for p in complete_model.parameters()) / 1e6
    backbone_params = sum(p.numel() for p in backbone.parameters()) / 1e6
    neck_params = sum(p.numel() for p in neck.parameters()) / 1e6
    head_params = sum(p.numel() for p in head.parameters()) / 1e6
    
    print(f"  Total Model: {total_params:.3f}M parameters")
    print(f"  Backbone:    {backbone_params:.3f}M parameters ({backbone_params/total_params*100:.1f}%)")
    print(f"  Neck:        {neck_params:.3f}M parameters ({neck_params/total_params*100:.1f}%)")
    print(f"  Head:        {head_params:.3f}M parameters ({head_params/total_params*100:.1f}%)")
    
    # Set models to evaluation mode
    backbone.eval()
    neck.eval()
    head.eval()
    complete_model.eval()
    
    print("\n⚡ Processing through model pipeline...")
    
    with torch.no_grad():
        # Step 1: Backbone
        print("  1. Backbone processing...")
        backbone_out1, backbone_out2, backbone_out3 = backbone(input_tensor)
        backbone_outputs = [backbone_out1, backbone_out2, backbone_out3]
        
        # Step 2: Neck  
        print("  2. Neck processing...")
        neck_out1, neck_out2, neck_out3 = neck(backbone_out1, backbone_out2, backbone_out3)
        neck_outputs = [neck_out1, neck_out2, neck_out3]
        
        # Step 3: Head (training mode)
        if test_training_mode:
            print("  3. Head processing (training mode)...")
            head.training = True
            head_training_outputs = head(neck_outputs.copy())
        
        # Step 4: Head (inference mode)
        print("  4. Head processing (inference mode)...")
        head.training = False
        head_inference_output = head(neck_outputs.copy())
        
        # Step 5: Complete model (should match head inference)
        print("  5. Complete model processing...")
        complete_output = complete_model(input_tensor)
    
    # Print detailed output information
    print(f"\n📊 Detailed Pipeline Outputs:")
    
    print(f"\n🏗️  Backbone Outputs:")
    for i, out in enumerate(backbone_outputs):
        downsampling = 2**(i+3)  # 8x, 16x, 32x
        print(f"  P{i+3}: {out.shape} - {downsampling}x downsampled")
    
    print(f"\n🌉 Neck Outputs:")
    for i, out in enumerate(neck_outputs):
        downsampling = 2**(i+3)  # 8x, 16x, 32x  
        print(f"  Enhanced P{i+3}: {out.shape} - {downsampling}x downsampled")
    
    if test_training_mode:
        print(f"\n🎯 Head Outputs (Training):")
        total_elements = 0
        for i, out in enumerate(head_training_outputs):
            elements = out.shape[2] * out.shape[3]  # H * W
            total_elements += elements
            print(f"  P{i+3}: {out.shape} - {elements} anchors - 64 box + 80 class channels")
        print(f"  Total training anchors: {total_elements}")
    
    print(f"\n🎯 Head Output (Inference):")
    print(f"  Final: {head_inference_output.shape}")
    print(f"    - {head_inference_output.shape[1]} total anchors")
    print(f"    - {head_inference_output.shape[2]} features (4 box coords + 80 classes)")
    
    print(f"\n✅ Complete Model Output:")
    print(f"  Shape: {complete_output.shape}")
    print(f"  Matches head inference: {torch.allclose(complete_output, head_inference_output, atol=1e-6)}")
    
    # Visualizations
    print(f"\n🎨 Generating visualizations...")
    
    # 1. Backbone features
    print("  📊 Backbone feature maps...")
    visualize_backbone_features(backbone_outputs, original_image)
    
    # 2. Neck features  
    print("  📊 Neck feature maps...")
    visualize_neck_features(neck_outputs)
    
    # 3. Head outputs
    print("  📊 Head outputs...")
    if test_training_mode:
        visualize_head_outputs(head_inference_output, head_training_outputs)
    else:
        visualize_head_outputs(head_inference_output)
    
    return {
        'backbone_outputs': backbone_outputs,
        'neck_outputs': neck_outputs, 
        'head_training_outputs': head_training_outputs if test_training_mode else None,
        'head_inference_output': head_inference_output,
        'complete_output': complete_output
    }

