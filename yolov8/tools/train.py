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
from yolov8.tools.utils import load_config, get_optimizer, get_scheduler # Ensure these are correctly named and present

# Placeholder for a proper loss function
# from yolov8.loss import ComputeLoss # We'll need to create this later

# Placeholder for evaluation metrics
# from yolov8.metrics import MeanAveragePrecision # Or similar

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

    # --- 3. Data Loading ---
    print("Setting up datasets and dataloaders...")
    dataset_cfg = cfg['dataset']
    model_cfg = cfg['model']
    img_h, img_w = model_cfg.get('input_size', [640, 640])

    # Basic transforms (more advanced augmentations from config to be added later)
    # TODO: Integrate augmentations from cfg['training']['augmentation']
    transform = T.Compose([
        T.Resize((img_h, img_w)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
    ])

    train_dataset = COCODataset(
        images_dir=dataset_cfg['train_images_path'],
        annotations_file=dataset_cfg['train_annotations_path'],
        transform=transform
    )
    # It's good practice to also have a validation set.
    val_dataset = COCODataset(
        images_dir=dataset_cfg['val_images_path'],
        annotations_file=dataset_cfg['val_annotations_path'],
        transform=transform # Usually validation uses simpler transforms (no heavy augmentation)
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
    
    # TODO: Add capability to load pretrained weights if specified in cfg['model']['pretrained_weights_path']
    # if model_cfg.get('pretrained_weights_path'):
    #     print(f"Loading pretrained weights from {model_cfg['pretrained_weights_path']}")
    #     model.load_state_dict(torch.load(model_cfg['pretrained_weights_path'], map_location=device))


    # --- 5. Optimizer and Scheduler ---
    print("Setting up optimizer and scheduler...")
    optimizer = get_optimizer(model.parameters(), cfg)
    scheduler = get_scheduler(optimizer, cfg)

    # --- 6. Loss Function ---
    # TODO: Initialize the actual loss function
    # criterion = ComputeLoss(model, cfg) # This will need the model for anchor/grid calculations
    print("Placeholder for Loss Function. Actual loss computation needed.")
    
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

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            # Targets are already collated: [batch_img_idx, class_label, x_center, y_center, w, h]
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(images) # Output format depends on model.train() vs model.eval()
                                        # Head in training mode returns list of feature maps

            # TODO: Compute loss
            # loss, loss_items = criterion(predictions, targets) 
            loss = torch.tensor(0.0, device=device, requires_grad=True) # Placeholder loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                # Consider stopping training or other error handling
                continue

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0: # Log every 10 batches
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Batch Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} Summary: Avg Loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s")

        # Learning rate scheduler step (if active)
        if scheduler:
            scheduler.step()

        # --- 8. Validation (Placeholder) ---
        # It's good practice to validate periodically
        if (epoch + 1) % cfg['training'].get('val_interval', 1) == 0: # Validate every 'val_interval' epochs
            print("Running validation (placeholder)...")
            # val_metric = validate_epoch(model, val_loader, device, cfg) # Implement this function
            # print(f"Validation Metric (placeholder): {val_metric}")
            # if val_metric > best_val_metric:
            #     best_val_metric = val_metric
            #     torch.save(model.state_dict(), os.path.join(weights_dir, 'best.pt'))
            #     print(f"Saved new best model to {os.path.join(weights_dir, 'best.pt')}")
            pass # Placeholder for actual validation logic

        # --- 9. Save Checkpoints ---
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
        print(f"Best model weights saved to {os.path.join(weights_dir, 'best.pt')}")


# Placeholder for validation function (to be implemented in eval.py or here)
# def validate_epoch(model, val_loader, device, cfg):
#     model.eval()
#     # ... validation logic ...
#     # Calculate mAP or other metrics
#     model.train()
#     return 0.0 # Placeholder metric

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