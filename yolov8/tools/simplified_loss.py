import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import complete_box_iou, box_iou

class SimplifiedYOLOLoss(nn.Module):
    """
    Simplified YOLO loss using reliable PyTorch implementations.
    This is much more reliable than implementing from scratch.
    """
    
    def __init__(self, num_classes, device, img_size=(640, 640), strides=[8, 16, 32], 
                 alpha=0.25, gamma=1.5, box_weight=7.5, cls_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.img_size = img_size
        self.strides = torch.tensor(strides, device=device)
        self.alpha = alpha
        self.gamma = gamma
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: List of prediction tensors from different scales
            targets: Ground truth targets [batch_idx, class, x, y, w, h] normalized
        """
        device = self.device
        total_loss = torch.tensor(0.0, device=device)
        loss_items = {}
        
        # Simple assignment strategy - match targets to closest anchors
        # This is simplified but more reliable than complex assignment
        batch_size = predictions[0].shape[0]
        
        for scale_idx, pred in enumerate(predictions):
            # pred shape: [batch, anchors*grid*grid, 4+num_classes]
            stride = self.strides[scale_idx]
            grid_size = int(self.img_size[0] / stride)
            
            # Extract predictions
            pred = pred.view(batch_size, -1, 4 + self.num_classes)
            pred_boxes = pred[..., :4]  # x,y,w,h
            pred_conf = pred[..., 4:4+self.num_classes]
            
            # Simple target assignment
            scale_targets = self._assign_targets_to_scale(targets, stride, grid_size, batch_size)
            
            if scale_targets is not None and len(scale_targets) > 0:
                target_boxes = scale_targets[:, 2:6]  # x,y,w,h
                target_classes = scale_targets[:, 1].long()
                target_indices = scale_targets[:, :2].long()  # batch_idx, anchor_idx
                
                # Ensure indices are within bounds
                valid_mask = (target_indices[:, 0] < batch_size) & (target_indices[:, 1] < pred_boxes.shape[1])
                
                if valid_mask.sum() > 0:
                    target_indices = target_indices[valid_mask]
                    target_boxes = target_boxes[valid_mask]
                    target_classes = target_classes[valid_mask]
                    
                    # Get corresponding predictions
                    pred_boxes_matched = pred_boxes[target_indices[:, 0], target_indices[:, 1]]
                    pred_conf_matched = pred_conf[target_indices[:, 0], target_indices[:, 1]]
                    
                    # Box loss using CIoU
                    if len(pred_boxes_matched) > 0:
                        iou_loss = self._box_loss(pred_boxes_matched, target_boxes)
                        
                        # Classification loss using focal loss
                        cls_loss = self._classification_loss(pred_conf_matched, target_classes)
                        
                        total_loss += self.box_weight * iou_loss + self.cls_weight * cls_loss
                        
                        loss_items[f'box_loss_scale_{scale_idx}'] = iou_loss.item()
                        loss_items[f'cls_loss_scale_{scale_idx}'] = cls_loss.item()
        
        return total_loss, loss_items
    
    def _assign_targets_to_scale(self, targets, stride, grid_size, batch_size):
        """Simple target assignment - assign to grid cell centers"""
        if len(targets) == 0:
            return None
            
        # Convert normalized coordinates to grid coordinates
        targets_scaled = targets.clone()
        targets_scaled[:, 2] *= grid_size  # x
        targets_scaled[:, 3] *= grid_size  # y
        targets_scaled[:, 4] *= grid_size  # w  
        targets_scaled[:, 5] *= grid_size  # h
        
        # Find which grid cell each target belongs to
        grid_x = torch.clamp(targets_scaled[:, 2].long(), 0, grid_size - 1)
        grid_y = torch.clamp(targets_scaled[:, 3].long(), 0, grid_size - 1)
        
        # Calculate anchor indices (simplified - just use grid position)
        # Make sure indices are within bounds
        max_anchors = grid_size * grid_size
        anchor_indices = torch.clamp(grid_y * grid_size + grid_x, 0, max_anchors - 1)
        
        # Create assignment tensor: [batch_idx, anchor_idx, class, x, y, w, h]
        assignments = torch.stack([
            targets_scaled[:, 0],  # batch_idx
            anchor_indices.float(),  # anchor_idx
            targets_scaled[:, 1],   # class
            targets_scaled[:, 2],   # x
            targets_scaled[:, 3],   # y
            targets_scaled[:, 4],   # w
            targets_scaled[:, 5],   # h
        ], dim=1)
        
        return assignments
    
    def _box_loss(self, pred_boxes, target_boxes):
        """Calculate box regression loss using CIoU"""
        # Convert to xyxy format for CIoU calculation
        pred_xyxy = self._xywh_to_xyxy(pred_boxes)
        target_xyxy = self._xywh_to_xyxy(target_boxes)
        
        # Use complete_box_iou for CIoU
        iou = complete_box_iou(pred_xyxy, target_xyxy)
        loss = 1 - iou.mean()
        
        return loss
    
    def _classification_loss(self, pred_conf, target_classes):
        """Calculate classification loss using focal loss"""
        # Create one-hot targets
        targets_one_hot = F.one_hot(target_classes, num_classes=self.num_classes).float()
        
        # Apply sigmoid to predictions
        pred_sigmoid = torch.sigmoid(pred_conf)
        
        # Manual focal loss implementation (more reliable than torchvision)
        ce_loss = F.binary_cross_entropy(pred_sigmoid, targets_one_hot, reduction='none')
        p_t = pred_sigmoid * targets_one_hot + (1 - pred_sigmoid) * (1 - targets_one_hot)
        alpha_t = self.alpha * targets_one_hot + (1 - self.alpha) * (1 - targets_one_hot)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = (focal_weight * ce_loss).mean()
        
        return loss
    
    def _xywh_to_xyxy(self, boxes):
        """Convert from center format to corner format"""
        x_center, y_center, width, height = boxes.unbind(-1)
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)


# Backwards compatibility - keep the old interface
def ComputeLoss(model_head=None, num_classes=80, device='cpu', img_size=(640, 640), 
                strides=[8, 16, 32], dfl_ch=16, reg_max=16, iou_type='ciou'):
    """
    Factory function to create the simplified loss function.
    Maintains backwards compatibility with existing code.
    """
    return SimplifiedYOLOLoss(
        num_classes=num_classes,
        device=device,
        img_size=img_size,
        strides=strides
    )

# Keep the bbox_iou function for validation
def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """Calculate IoU using torchvision's reliable implementation"""
    if xywh:
        # Convert from center format to corner format
        box1_xyxy = torch.cat((box1[..., :2] - box1[..., 2:] / 2,
                               box1[..., :2] + box1[..., 2:] / 2), dim=-1)
        box2_xyxy = torch.cat((box2[..., :2] - box2[..., 2:] / 2,
                               box2[..., :2] + box2[..., 2:] / 2), dim=-1)
    else:
        box1_xyxy = box1
        box2_xyxy = box2
    
    if CIoU:
        return complete_box_iou(box1_xyxy, box2_xyxy)
    else:
        return box_iou(box1_xyxy, box2_xyxy)
