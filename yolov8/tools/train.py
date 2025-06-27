import argparse
import os
import time
import yaml # Already imported but good to ensure
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T # For basic transforms

# Relative imports from the yolov8 package
from yolov8.tools.dataset import COCODataset
from yolov8.yolov8 import YOLOv8
from yolov8.tools.utils import load_config, get_optimizer, get_scheduler
from yolov8.tools.loss import ComputeLoss, bbox_iou # Import bbox_iou for potential use in validation
from torchvision.ops import nms # For NMS in validation
from torch.utils.tensorboard import SummaryWriter # For TensorBoard logging

# Placeholder for evaluation metrics
# from yolov8.metrics import MeanAveragePrecision # Or similar

@torch.no_grad() # Decorator for functions that don't need gradient calculation
def validate_epoch(model, val_loader, device, cfg, epoch_num=-1):
    """
    Validates the model on the validation set for one epoch.
    Args:
        model: The YOLOv8 model.
        val_loader: DataLoader for the validation set.
        device: Device to run validation on (cuda/cpu).
        cfg: Configuration dictionary.
        epoch_num (int): Current epoch number, for logging.
    Returns:
        (float): Placeholder for mAP or other primary validation metric.
                 Currently returns average number of detections per image.
    """
    print(f"\n--- Validating Epoch {epoch_num if epoch_num > 0 else ''} ---")
    model.eval() # Set model to evaluation mode

    eval_cfg = cfg['evaluation']
    conf_thresh = eval_cfg.get('confidence_threshold', 0.25)
    iou_thresh_nms = eval_cfg.get('iou_threshold', 0.45) # NMS IoU threshold
    model_input_h, model_input_w = cfg['model'].get('input_size', [640, 640])

    all_predictions = [] # To store predictions for mAP calculation
    all_targets = []     # To store ground truths for mAP calculation
    
    total_detections = 0
    num_images_processed = 0

    for batch_idx, (images, targets) in enumerate(val_loader):
        images = images.to(device)
        # Targets are [batch_img_idx, class_label, x_center_norm, y_center_norm, w_norm, h_norm]
        # We need to prepare targets per image for matching with predictions later.
        
        batch_size = images.shape[0]
        num_images_processed += batch_size

        raw_predictions = model(images) # Output: [B, num_preds, 4_coords + num_classes]

        # Process predictions for each image in the batch
        for i in range(batch_size):
            pred_single = raw_predictions[i] # [num_preds, 4_coords + num_classes]
            
            # Extract boxes (cx,cy,w,h absolute on model input scale), scores, class_indices
            boxes_cxcywh = pred_single[:, :4]
            x_center, y_center, width, height = boxes_cxcywh.T
            boxes_xyxy = torch.stack((x_center - width / 2, y_center - height / 2,
                                      x_center + width / 2, y_center + height / 2), dim=1)
            
            scores, class_indices = torch.max(pred_single[:, 4:], dim=1)

            # Apply confidence threshold
            conf_mask = scores > conf_thresh
            boxes_after_conf = boxes_xyxy[conf_mask]
            scores_after_conf = scores[conf_mask]
            class_indices_after_conf = class_indices[conf_mask]

            # NMS per class
            img_final_boxes = []
            img_final_scores = []
            img_final_labels = []

            unique_classes = torch.unique(class_indices_after_conf)
            for cls_idx in unique_classes:
                cls_mask = class_indices_after_conf == cls_idx
                cls_boxes = boxes_after_conf[cls_mask]
                cls_scores = scores_after_conf[cls_mask]
                
                if cls_boxes.shape[0] == 0: continue

                keep = nms(cls_boxes, cls_scores, iou_thresh_nms)
                img_final_boxes.append(cls_boxes[keep])
                img_final_scores.append(cls_scores[keep])
                img_final_labels.append(torch.full_like(cls_scores[keep], fill_value=cls_idx.item(), dtype=torch.long))
            
            if img_final_boxes:
                img_final_boxes = torch.cat(img_final_boxes, dim=0)
                img_final_scores = torch.cat(img_final_scores, dim=0)
                img_final_labels = torch.cat(img_final_labels, dim=0)
                total_detections += img_final_boxes.shape[0]
            else: # Ensure empty tensors if no detections
                img_final_boxes = torch.empty(0, 4, device=device)
                img_final_scores = torch.empty(0, device=device)
                img_final_labels = torch.empty(0, dtype=torch.long, device=device)

            # Store predictions for this image (for mAP)
            # Format: dict with 'boxes', 'scores', 'labels' (all tensors)
            # Boxes should be in xyxy format, absolute pixel coords (model input scale)
            all_predictions.append({
                'boxes': img_final_boxes, # Already xyxy, model input scale
                'scores': img_final_scores,
                'labels': img_final_labels
            })

            # Store ground truths for this image (for mAP)
            # Targets are normalized [cls, cx, cy, w, h]. Convert to xyxy absolute pixel coords.
            gt_for_img_mask = targets[:, 0] == i
            gt_for_img_norm = targets[gt_for_img_mask][:, 1:] # [cls, cx, cy, w, h] normalized
            
            gt_boxes_xyxy_abs = torch.empty(0, 4, device=device)
            gt_labels_abs = torch.empty(0, dtype=torch.long, device=device)

            if gt_for_img_norm.shape[0] > 0:
                gt_cls = gt_for_img_norm[:, 0].long()
                gt_cxcywh_norm = gt_for_img_norm[:, 1:]
                
                # Denormalize
                gt_cx_abs = gt_cxcywh_norm[:, 0] * model_input_w
                gt_cy_abs = gt_cxcywh_norm[:, 1] * model_input_h
                gt_w_abs = gt_cxcywh_norm[:, 2] * model_input_w
                gt_h_abs = gt_cxcywh_norm[:, 3] * model_input_h
                
                gt_x1_abs = gt_cx_abs - gt_w_abs / 2
                gt_y1_abs = gt_cy_abs - gt_h_abs / 2
                gt_x2_abs = gt_cx_abs + gt_w_abs / 2
                gt_y2_abs = gt_cy_abs + gt_h_abs / 2
                
                gt_boxes_xyxy_abs = torch.stack([gt_x1_abs, gt_y1_abs, gt_x2_abs, gt_y2_abs], dim=1)
                gt_labels_abs = gt_cls

            all_targets.append({
                'boxes': gt_boxes_xyxy_abs, # xyxy, model input scale
                'labels': gt_labels_abs
            })
            
        if (batch_idx + 1) % 10 == 0 :
             print(f"  Validated batch {batch_idx+1}/{len(val_loader)}")

    # --- mAP Calculation Placeholder ---
    # A full mAP calculation is complex. For now, this is a placeholder.
    # It would involve:
    # 1. For each class:
    #    a. Collect all predicted boxes, scores for that class.
    #    b. Collect all GT boxes for that class.
    #    c. Match predictions to GTs based on IoU threshold (e.g., 0.5).
    #    d. Calculate precision-recall curve.
    #    e. Calculate Average Precision (AP) for the class.
    # 2. mAP is the mean of APs over all classes.
    # Libraries like `torchmetrics.detection.MeanAveragePrecision` can do this.
    
    # Placeholder metric: average number of detections per image
    avg_detections_per_image = total_detections / num_images_processed if num_images_processed > 0 else 0.0
    map_placeholder = avg_detections_per_image # Replace with actual mAP later

    print(f"--- Validation Summary ---")
    print(f"Processed {num_images_processed} images.")
    print(f"Total Detections (after NMS & conf_thresh): {total_detections}")
    print(f"Average Detections per Image: {avg_detections_per_image:.2f}")
    print(f"mAP (Placeholder - using avg detections): {map_placeholder:.4f}")
    print(f"TODO: Implement proper mAP calculation using 'all_predictions' and 'all_targets'.")
    
    model.train() # Set model back to training mode
    return map_placeholder # Return the main metric

def train(config_path):
    """
    Main training script for YOLOv8.
    """
    # --- 1. Load Configuration ---
    cfg = load_config(config_path)
    print(f"Configuration loaded from {config_path}")

    # --- 2. Setup ---
    # Device
    device_str = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("CUDA specified but not available. Falling back to CPU.")
        device_str = 'cpu'
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Seed for reproducibility
    seed = cfg['training'].get('seed', 42)
    torch.manual_seed(seed)
    if device_str == 'cuda':
        torch.cuda.manual_seed(seed)
    # np.random.seed(seed) # If using numpy for augmentations not handled by torch

    # Directories for output
    exp_name = cfg['training'].get('experiment_name', 'exp')
    output_dir = os.path.join(cfg['training'].get('log_dir', 'runs/train'), exp_name)
    weights_dir = os.path.join(output_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")
    
    # Save a copy of the config used for this run
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)
    
    # --- TensorBoard Writer ---
    tb_log_dir = os.path.join(output_dir, 'tensorboard_logs')
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logs will be saved to: {tb_log_dir}")

    # --- 3. Data Loading ---
    print("Setting up datasets and dataloaders...")
    dataset_cfg = cfg['dataset']
    model_cfg = cfg['model']
    img_h, img_w = model_cfg.get('input_size', [640, 640])

    # Augmentation parameters from config
    augmentation_params = cfg['training'].get('augmentation', {})

    train_dataset = COCODataset(
        images_dir=dataset_cfg['train_images_path'],
        annotations_file=dataset_cfg['train_annotations_path'],
        transform_params=augmentation_params,
        is_train=True,
        img_size=(img_h, img_w),
        num_classes=dataset_cfg['num_classes']
    )
    # For validation, typically no heavy augmentations, only resize, normalize, ToTensor.
    # COCODataset handles this with is_train=False.
    val_dataset = COCODataset(
        images_dir=dataset_cfg['val_images_path'],
        annotations_file=dataset_cfg['val_annotations_path'],
        transform_params={}, # Pass empty dict for val, or specific val_augmentations if defined
        is_train=False,
        img_size=(img_h, img_w),
        num_classes=dataset_cfg['num_classes']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg.get('workers', 4),
        collate_fn=train_dataset.collate_fn, # Important for handling variable targets
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['evaluation'].get('batch_size', cfg['training']['batch_size'] * 2),
        shuffle=False,
        num_workers=cfg.get('workers', 4),
        collate_fn=val_dataset.collate_fn,
        pin_memory=True
    )
    print(f"Train dataset: {len(train_dataset)} samples, Val dataset: {len(val_dataset)} samples")

    # --- 4. Model Initialization ---
    print("Initializing model...")
    model = YOLOv8(
        version=model_cfg['architecture'],
        num_classes=dataset_cfg['num_classes'],
        # dfl_ch can be added to model_cfg if we want to configure it
        # dfl_ch=model_cfg.get('dfl_channels', 16) 
    ).to(device)
    
    # Load pretrained weights if specified
    pretrained_weights_path = model_cfg.get('pretrained_weights_path')
    if pretrained_weights_path:
        if os.path.exists(pretrained_weights_path):
            print(f"Loading pretrained weights from {pretrained_weights_path}")
            try:
                state_dict = torch.load(pretrained_weights_path, map_location=device)
                # Handle potential 'module.' prefix if saved with DataParallel
                if any(key.startswith('module.') for key in state_dict.keys()):
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] if k.startswith('module.') else k 
                        new_state_dict[name] = v
                    model.load_state_dict(new_state_dict, strict=False) # strict=False to allow for mismatches (e.g. different num_classes)
                else:
                    model.load_state_dict(state_dict, strict=False)
                print("Successfully loaded pretrained weights.")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}. Training from scratch.")
        else:
            print(f"Warning: Pretrained weights path specified but not found: {pretrained_weights_path}. Training from scratch.")
    else:
        print("No pretrained weights specified. Training from scratch.")


    # Ensure model head has correct strides (critical for loss calculation and anchor generation)
    default_strides = [8., 16., 32.] # TODO: Get this from config if possible
    if hasattr(model, 'head') and hasattr(model.head, 'stride'):
        if torch.all(model.head.stride == 0).item() or model.head.stride.shape[0] != len(default_strides):
            print(f"Warning: Model head strides are not set or incorrect. Setting to default: {default_strides}")
            model.head.stride = torch.tensor(default_strides, device=device)
        else:
            # Ensure strides are on the correct device if already set
            model.head.stride = model.head.stride.to(device)
            print(f"Model head strides: {model.head.stride.tolist()}")
    else:
        print("Warning: Model head or head.stride not found. Loss function will use default strides.")
        # ComputeLoss takes strides as an argument, so this is okay.

    # --- 5. Optimizer and Scheduler ---
    print("Setting up optimizer and scheduler...")
    optimizer = get_optimizer(model.parameters(), cfg)
    scheduler = get_scheduler(optimizer, cfg)

    # --- 6. Loss Function ---
    print("Initializing Loss Function...")
    loss_cfg = cfg.get('loss', {}) 
    
    # Determine DFL channels (reg_max in YOLOv8 context often means dfl_channels)
    # Head's `self.ch` is DFL channels.
    dfl_ch_from_head = getattr(getattr(model, 'head', None), 'ch', 16) 
    
    criterion = ComputeLoss(
        model_head=getattr(model, 'head', None), 
        num_classes=dataset_cfg['num_classes'],
        device=device,
        img_size=(img_h, img_w), 
        strides=default_strides, # Pass strides explicitly
        dfl_ch=dfl_ch_from_head, 
        reg_max=dfl_ch_from_head, # reg_max for DFL, typically same as dfl_ch
        iou_type=loss_cfg.get('iou_type', 'ciou'),
    )
    print(f"Loss function initialized with img_size=({img_h},{img_w}), strides={default_strides}, dfl_ch={dfl_ch_from_head}.")
    
    # --- 7. Training Loop ---
    epochs = cfg['training']['epochs']
    save_period = cfg['training'].get('save_period', 5) # Save every N epochs
    best_val_metric = float('-inf') # Or float('inf') if minimizing a metric like loss

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        print(f"--- Epoch {epoch+1}/{epochs} ---")
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6e}")
        if tb_writer: tb_writer.add_scalar('Training/Learning_Rate', current_lr, epoch)

        iter_count_epoch = 0 # For per-iteration logging if desired
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            # Targets are already collated: [batch_img_idx, class_label, x_center, y_center, w, h]
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            assert model.training, "Model should be in training mode." # Ensure model is in train mode
            predictions_from_head = model(images) 

            # Compute loss
            loss, loss_items = criterion(predictions_from_head, targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss detected at epoch {epoch+1}, batch {batch_idx}. Loss items: {loss_items}. Skipping batch.")
                # Consider stopping training or other error handling
                continue

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate weighted loss components for logging if needed
            # For now, epoch_loss sums the total loss from criterion
            epoch_loss += loss.item() # loss is already the total weighted loss

            if (batch_idx + 1) % 10 == 0: # Log every 10 batches
                log_str = f"  Batch {batch_idx+1}/{len(train_loader)}, Total Loss: {loss.item():.4f}"
                if loss_items:
                    log_str += f" (Box: {loss_items.get('loss_box', 0):.4f}, Cls: {loss_items.get('loss_cls', 0):.4f}, DFL: {loss_items.get('loss_dfl', 0):.4f})"
                print(log_str)
            
            # Log batch losses to TensorBoard
            if tb_writer and loss_items:
                global_step = epoch * len(train_loader) + batch_idx
                tb_writer.add_scalar('Loss/Batch/Total', loss.item(), global_step)
                tb_writer.add_scalar('Loss/Batch/Box', loss_items.get('loss_box', 0), global_step)
                tb_writer.add_scalar('Loss/Batch/Cls', loss_items.get('loss_cls', 0), global_step)
                tb_writer.add_scalar('Loss/Batch/DFL', loss_items.get('loss_dfl', 0), global_step)
        
        avg_epoch_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} Summary: Avg Total Loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s")
        if tb_writer:
            tb_writer.add_scalar('Loss/Epoch/Total', avg_epoch_loss, epoch)
            # TODO: Could accumulate and log average epoch component losses too if needed

        # Learning rate scheduler step (if active)
        if scheduler:
            scheduler.step()

        # --- 8. Validation ---
        if (epoch + 1) % cfg['training'].get('val_interval', 1) == 0 and val_loader:
            val_metric = validate_epoch(model, val_loader, device, cfg, epoch_num=epoch + 1)
            print(f"Epoch {epoch+1} Validation Metric (Placeholder mAP): {val_metric:.4f}")
            if tb_writer: tb_writer.add_scalar('Validation/mAP_Placeholder', val_metric, epoch)

            # Save best model based on validation metric
            if val_metric > best_val_metric: # Assuming higher is better for mAP
                best_val_metric = val_metric
                best_checkpoint_path = os.path.join(weights_dir, 'best.pt')
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"Saved new best model to {best_checkpoint_path} (Metric: {best_val_metric:.4f})")
        
        # --- 9. Save Checkpoints ---
        # Save epoch checkpoint
        if (epoch + 1) % save_period == 0 or (epoch + 1) == epochs:
            checkpoint_path = os.path.join(weights_dir, f'epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save last checkpoint
        last_checkpoint_path = os.path.join(weights_dir, 'last.pt')
        torch.save(model.state_dict(), last_checkpoint_path)


    print("Training finished.")
    print(f"Final model weights saved to {last_checkpoint_path}")
    if os.path.exists(os.path.join(weights_dir, 'best.pt')):
        print(f"Best model weights (based on val metric) saved to {os.path.join(weights_dir, 'best.pt')}")
    else:
        print(f"No 'best.pt' model saved (either validation was not run, or metric did not improve over initial {float('-inf')}).")

    if tb_writer:
        tb_writer.close()
        print("TensorBoard writer closed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLOv8 model.")
    parser.add_argument('--config', type=str, default='config/coco_yolov8.yaml', 
                        help='Path to the YAML configuration file.')
    # Potentially add overrides for specific config parameters here
    # parser.add_argument('--batch_size', type=int, help='Override batch size from config.')
    
    args = parser.parse_args()

    try:
        train(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure the config file path is correct and dataset paths within the config are valid.")
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        import traceback
        traceback.print_exc()