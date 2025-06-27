import argparse
import os
import cv2 # For image loading, drawing, and saving
import torch
import torchvision.transforms as T
from torchvision.ops import nms # Import NMS
from PIL import Image # Alternative image loading
import numpy as np
import glob # For finding image files
from typing import cast
import json # For saving detections

# Relative imports
from yolov8.yolov8 import YOLOv8
from yolov8.tools.utils import load_config


# Helper function to draw detections on an image
def draw_detections(image, boxes, scores, class_indices, class_names, conf_thresh=0.5):
    """
    Draws bounding boxes, class names, and scores on an image.
    Args:
        image (np.ndarray): Image in BGR format (OpenCV default).
        boxes (torch.Tensor or np.ndarray): [N, 4] tensor of bounding boxes (x1, y1, x2, y2).
        scores (torch.Tensor or np.ndarray): [N] tensor of scores.
        class_indices (torch.Tensor or np.ndarray): [N] tensor of class indices.
        class_names (list): List of class names.
        conf_thresh (float): Confidence threshold to filter detections.
    Returns:
        np.ndarray: Image with detections drawn.
    """
    if isinstance(image, torch.Tensor): # If image is a tensor, convert to numpy
        image = image.permute(1, 2, 0).cpu().numpy()
        if image.max() <= 1.0: # Denormalize if normalized
            image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Ensure BGR for OpenCV
    elif isinstance(image, Image.Image): # If PIL image
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    for i in range(boxes.shape[0]):
        score = scores[i]
        if score < conf_thresh:
            continue

        box = boxes[i]
        class_idx = int(class_indices[i])
        
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[class_idx]}: {score:.2f}"

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box

        # Put label text
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Black text

    return image

@torch.no_grad()
def test(config_path, checkpoint_path, source_path, output_dir="runs/detect/exp", conf_thresh=0.25, iou_thresh_nms=0.45):
    """
    Runs inference with a trained YOLOv8 model on images or a directory.
    """
    # --- 1. Load Configuration ---
    cfg = load_config(config_path)
    print(f"Configuration loaded from {config_path}")

    # --- 2. Setup ---
    device_str = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    # --- 3. Model Initialization and Loading Checkpoint ---
    model_cfg = cfg['model']
    dataset_cfg = cfg['dataset'] # For class names
    class_names = dataset_cfg.get('class_names', [f'class_{i}' for i in range(dataset_cfg['num_classes'])])

    print("Initializing model...")
    model = YOLOv8(
        version=model_cfg['architecture'],
        num_classes=dataset_cfg['num_classes'],
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading model weights from {checkpoint_path}...")
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except RuntimeError: # Handle 'module.' prefix if saved with DataParallel
        state_dict = torch.load(checkpoint_path, map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("Successfully loaded weights with key matching for DataParallel.")
    
    model.eval()

    # Set model head strides if necessary (as discussed in eval.py)
    if hasattr(model, 'head') and hasattr(model.head, 'stride') and torch.all(model.head.stride == 0).item():
        model.head.stride = torch.tensor([8., 16., 32.], device=device) # Common strides for P3,P4,P5

    # --- 4. Prepare Input Transform ---
    img_h, img_w = model_cfg.get('input_size', [640, 640])
    transform = T.Compose([
        T.Resize((img_h, img_w)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 5. Process Source Path ---
    image_paths = []
    if os.path.isdir(source_path):
        # Common image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(source_path, ext)))
        if not image_paths:
            print(f"No images found in directory: {source_path}")
            return
        print(f"Found {len(image_paths)} images in {source_path}")
    elif os.path.isfile(source_path):
        image_paths.append(source_path)
    else:
        raise FileNotFoundError(f"Source path not found or is not a file/directory: {source_path}")

    # --- 6. Inference Loop ---
    for img_path in image_paths:
        print(f"\nProcessing {img_path}...")
        try:
            # Load image using PIL (consistent with torchvision transforms) then convert to OpenCV
            pil_image = Image.open(img_path).convert('RGB')

            img_tensor = cast(torch.Tensor, transform(pil_image))
            img_tensor = img_tensor.unsqueeze(0).to(device) # Add batch dimension
            
            # For drawing, load with OpenCV to keep original for drawing
            cv_image_orig = cv2.imread(img_path)
            if cv_image_orig is None:
                print(f"Warning: OpenCV could not read {img_path}. Skipping.")
                continue
            
            original_h, original_w = cv_image_orig.shape[:2]

        except Exception as e:
            print(f"Error loading or transforming image {img_path}: {e}")
            continue

        # Perform inference
        predictions_raw = model(img_tensor) # Expected: [1, num_anchors, 4_coords + num_classes]

        # --- CRITICAL: Post-processing (NMS) ---
        # This part is highly dependent on a robust NMS implementation.
        # Placeholder: simple confidence thresholding and extraction. This is NOT NMS.
        # `predictions_raw` for a single image in batch: [num_anchors, 4+nc]
        pred_single = predictions_raw[0] 
        boxes_cxcywh_norm_or_abs = pred_single[:, :4] # Model head output: cx, cy, w, h (relative to input_size or normalized)
        
        # Assuming head output is absolute cx,cy,w,h for the input size (e.g., 640x640)
        # And not normalized to 0-1. If normalized, multiply by img_w, img_h.
        # Convert cx,cy,w,h to x1,y1,x2,y2 for NMS and drawing
        x_center, y_center, width, height = boxes_cxcywh_norm_or_abs.T
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        boxes_xyxy_model_input_scale = torch.stack((x1, y1, x2, y2), dim=1)

        scores, class_indices = torch.max(pred_single[:, 4:], dim=1)
        
        # Apply confidence threshold
        conf_mask = scores > conf_thresh
        boxes_after_conf = boxes_xyxy_model_input_scale[conf_mask]
        scores_after_conf = scores[conf_mask]
        class_indices_after_conf = class_indices[conf_mask]

        # Perform NMS per class
        final_boxes_list = []
        final_scores_list = []
        final_class_indices_list = []

        unique_classes = torch.unique(class_indices_after_conf)
        for cls_idx in unique_classes:
            cls_mask = class_indices_after_conf == cls_idx
            cls_boxes = boxes_after_conf[cls_mask]
            cls_scores = scores_after_conf[cls_mask]
            
            if cls_boxes.shape[0] == 0:
                continue

            # Apply NMS
            keep = nms(cls_boxes, cls_scores, iou_thresh_nms)
            
            final_boxes_list.append(cls_boxes[keep])
            final_scores_list.append(cls_scores[keep])
            # Create class indices tensor for the kept boxes, filled with current cls_idx
            final_class_indices_list.append(torch.full_like(cls_scores[keep], fill_value=cls_idx.item(), dtype=torch.long))

        if not final_boxes_list: # No detections after NMS
            print("No detections after NMS.")
            # Create empty tensors with consistent device
            final_boxes_model_scale = torch.empty(0, 4, device=device)
            final_scores_tensor = torch.empty(0, device=device)
            final_class_indices_tensor = torch.empty(0, dtype=torch.long, device=device)
        else:
            final_boxes_model_scale = torch.cat(final_boxes_list, dim=0)
            final_scores_tensor = torch.cat(final_scores_list, dim=0)
            final_class_indices_tensor = torch.cat(final_class_indices_list, dim=0)

        # Scale boxes from model input size (e.g., 640x640) to original image size
        if final_boxes_model_scale.shape[0] > 0:
            scale_x = original_w / img_w
            scale_y = original_h / img_h
            
            final_boxes_abs = final_boxes_model_scale.clone()
            final_boxes_abs[:, 0] *= scale_x # x1
            final_boxes_abs[:, 1] *= scale_y # y1
            final_boxes_abs[:, 2] *= scale_x # x2
            final_boxes_abs[:, 3] *= scale_y # y2
        else:
            final_boxes_abs = torch.empty(0, 4, device=device) # Ensure it's a tensor even if empty
        
        print(f"Found {final_boxes_abs.shape[0]} detections after NMS.")

        # Draw detections on the original OpenCV image
        # The draw_detections function has its own confidence threshold,
        # but since we already filtered by conf_thresh before NMS, 
        # we can pass 0.0 to draw_detections to draw all NMS survivors.
        output_image = draw_detections(
            cv_image_orig.copy(), # Draw on a copy
            final_boxes_abs.cpu().numpy(), 
            final_scores_tensor.cpu().numpy(), 
            final_class_indices_tensor.cpu().numpy(),
            class_names,
            conf_thresh=0.0 # Draw all boxes that survived NMS and initial conf_thresh
        )

        # Save the output image
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        output_image_path = os.path.join(output_dir, f"{base_filename}_detected.jpg")
        cv2.imwrite(output_image_path, output_image)
        print(f"Saved visual detection to {output_image_path}")

        # Save detections as JSON
        detections_json_path = os.path.join(output_dir, f"{base_filename}_detections.json")
        output_detections = []
        if final_boxes_abs.shape[0] > 0:
            for i in range(final_boxes_abs.shape[0]):
                box = final_boxes_abs[i].cpu().tolist() # [x1, y1, x2, y2]
                score = final_scores_tensor[i].cpu().item()
                class_id = final_class_indices_tensor[i].cpu().item()
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                
                output_detections.append({
                    'box_xyxy': [round(coord, 2) for coord in box],
                    'score': round(score, 4),
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        with open(detections_json_path, 'w') as f:
            json.dump(output_detections, f, indent=4)
        print(f"Saved JSON detections to {detections_json_path}")


    print("\nTesting finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on images.")
    parser.add_argument('--config', type=str, default='config/coco_yolov8.yaml',
                        help='Path to the YAML configuration file.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained model checkpoint (.pt file).')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to an image file or a directory of images.')
    parser.add_argument('--output_dir', type=str, default='runs/detect/exp',
                        help='Directory to save output images with detections.')
    parser.add_argument('--conf_thresh', type=float, default=0.25,
                        help='Confidence threshold for detections.')
    parser.add_argument('--iou_thresh_nms', type=float, default=0.45,
                        help='IoU threshold for Non-Maximum Suppression (currently placeholder).')

    args = parser.parse_args()

    try:
        test(args.config, args.checkpoint, args.source, args.output_dir, args.conf_thresh, args.iou_thresh_nms)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check file paths.")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
