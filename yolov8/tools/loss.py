import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou # For IoU and GIoU/CIoU

# Placeholder for a more advanced assigner like TaskAlignedAssigner
# For now, we might use a simpler assignment or assume one is provided/integrated.

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate IoU, GIoU, DIoU, or CIoU.
    Args:
        box1 (torch.Tensor): Predicted bboxes, shape (N, 4) or (bs, N, 4).
        box2 (torch.Tensor): Ground truth bboxes, shape (M, 4) or (bs, M, 4).
        xywh (bool): If True, boxes are in [x_center, y_center, w, h] format,
                     otherwise [x1, y1, x2, y2].
        GIoU (bool): If True, calculate GIoU.
        DIoU (bool): If True, calculate DIoU.
        CIoU (bool): If True, calculate CIoU.
        eps (float): Small value to prevent division by zero.
    Returns:
        (torch.Tensor): IoU/GIoU/DIoU/CIoU values, shape (N, M) or (bs, N, M).
    """
    # Convert xywh to xyxy if needed
    if xywh:
        box1_xyxy = torch.cat((box1[..., :2] - box1[..., 2:] / 2,
                               box1[..., :2] + box1[..., 2:] / 2), dim=-1)
        box2_xyxy = torch.cat((box2[..., :2] - box2[..., 2:] / 2,
                               box2[..., :2] + box2[..., 2:] / 2), dim=-1)
    else:
        box1_xyxy = box1
        box2_xyxy = box2

    # Ensure correct device and dtype
    box1_xyxy = box1_xyxy.to(box2_xyxy.device, non_blocking=True).float()
    box2_xyxy = box2_xyxy.to(box2_xyxy.device, non_blocking=True).float()


    # Intersection
    inter_x1 = torch.max(box1_xyxy[..., 0], box2_xyxy[..., 0])
    inter_y1 = torch.max(box1_xyxy[..., 1], box2_xyxy[..., 1])
    inter_x2 = torch.min(box1_xyxy[..., 2], box2_xyxy[..., 2])
    inter_y2 = torch.min(box1_xyxy[..., 3], box2_xyxy[..., 3])
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    # Union
    box1_area = (box1_xyxy[..., 2] - box1_xyxy[..., 0]) * (box1_xyxy[..., 3] - box1_xyxy[..., 1])
    box2_area = (box2_xyxy[..., 2] - box2_xyxy[..., 0]) * (box2_xyxy[..., 3] - box2_xyxy[..., 1])
    union_area = box1_area + box2_area - inter_area + eps
    iou = inter_area / union_area

    if GIoU or DIoU or CIoU:
        # Enclosing box
        enclose_x1 = torch.min(box1_xyxy[..., 0], box2_xyxy[..., 0])
        enclose_y1 = torch.min(box1_xyxy[..., 1], box2_xyxy[..., 1])
        enclose_x2 = torch.max(box1_xyxy[..., 2], box2_xyxy[..., 2])
        enclose_y2 = torch.max(box1_xyxy[..., 3], box2_xyxy[..., 3])
        enclose_w = torch.clamp(enclose_x2 - enclose_x1, min=0)
        enclose_h = torch.clamp(enclose_y2 - enclose_y1, min=0)
        enclose_area = enclose_w * enclose_h + eps

        if GIoU:
            return iou - (enclose_area - union_area) / enclose_area
        
        # DIoU or CIoU
        # Distance between centers
        b1_cx = (box1_xyxy[..., 0] + box1_xyxy[..., 2]) / 2
        b1_cy = (box1_xyxy[..., 1] + box1_xyxy[..., 3]) / 2
        b2_cx = (box2_xyxy[..., 0] + box2_xyxy[..., 2]) / 2
        b2_cy = (box2_xyxy[..., 1] + box2_xyxy[..., 3]) / 2
        center_dist_sq = (b1_cx - b2_cx)**2 + (b1_cy - b2_cy)**2
        
        # Diagonal of enclosing box
        diag_enclose_sq = enclose_w**2 + enclose_h**2
        
        diou_term = center_dist_sq / diag_enclose_sq
        if DIoU:
            return iou - diou_term

        if CIoU:
            # Aspect ratio consistency
            arctan = torch.atan # to ensure it's torch.atan
            w1, h1 = box1_xyxy[..., 2] - box1_xyxy[..., 0], box1_xyxy[..., 3] - box1_xyxy[..., 1]
            w2, h2 = box2_xyxy[..., 2] - box2_xyxy[..., 0], box2_xyxy[..., 3] - box2_xyxy[..., 1]
            v = (4 / (torch.pi**2)) * torch.pow(arctan(w2 / (h2 + eps)) - arctan(w1 / (h1 + eps)), 2)
            alpha = v / (1 - iou + v + eps) # Detach alpha to prevent backprop through it for v
            alpha = alpha.detach() 
            return iou - diou_term - alpha * v
    return iou


class ComputeLoss(nn.Module):
    def __init__(self, model_head, num_classes, device, img_size, strides=[8., 16., 32.], dfl_ch=16,
                 reg_max=16, iou_type='ciou', bce_pos_weight=None):
        super().__init__()
        self.model_head = model_head # For accessing parts like make_anchors if needed, or dfl
        self.num_classes = num_classes
        self.device = device
        self.img_size_h = img_size[0] 
        self.img_size_w = img_size[1]
        self.strides = torch.tensor(strides, device=device)
        self.dfl_ch = dfl_ch # Number of channels for DFL per coordinate (e.g., 16)
        self.reg_max = reg_max # Max value for DFL regression, typically dfl_ch / 2 or dfl_ch -1
                               # This defines the range [0, reg_max-1] for DFL.
                               # The head's DFL module uses self.ch (dfl_ch) directly. Let's assume reg_max = dfl_ch for now.

        self.iou_type = iou_type.lower()
        assert self.iou_type in ['iou', 'giou', 'diou', 'ciou'], f"Unsupported iou_type: {iou_type}"

        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight, reduction='none')
        # For DFL, the loss is typically a cross-entropy loss on the distribution.
        # Each coordinate (l, t, r, b) is predicted as a distribution over 'reg_max' bins.
        # The ground truth is a one-hot (or two-hot for values between bins) encoding of the target coordinate.

        # Loss weights (can be tuned or loaded from config)
        self.lambda_box = 7.5  # Corresponds to box_loss_gain in some YOLO versions
        self.lambda_cls = 0.5  # Corresponds to cls_loss_gain
        self.lambda_dfl = 1.5  # Corresponds to dfl_loss_gain

        # Assigner (placeholder, will need a proper one like TaskAlignedAssigner)
        # For now, we'll implement a simplified assignment within the forward pass.
        self.assigner = self.default_assigner 
        self.decode_bboxes_fn = self.decode_dfl_bboxes if self.dfl_ch > 0 else self.decode_simple_bboxes

    def decode_dfl_bboxes(self, pred_dist, anchors_xywh):
        """
        Decodes bounding boxes from DFL distribution predictions.
        pred_dist: (N, 4 * dfl_ch) or (N, 4, dfl_ch)
        anchors_xywh: (N, 4) in [cx, cy, w, h] format (absolute, not normalized by stride yet)
        Returns: (N, 4) decoded boxes in [cx, cy, w, h] format (absolute)
        """
        if pred_dist.ndim == 2:
            pred_dist = pred_dist.view(-1, 4, self.dfl_ch) # (N, 4, dfl_ch)
        
        # Softmax over the distribution for each coordinate part
        pred_dist_softmax = F.softmax(pred_dist, dim=2) # (N, 4, dfl_ch)
        
        # Create the integration range [0, 1, ..., reg_max-1]
        # self.reg_max should be self.dfl_ch for this simple integration
        integration_range = torch.arange(self.dfl_ch, device=pred_dist.device, dtype=pred_dist.dtype).float() # (dfl_ch,)
        
        # Multiply softmax probabilities by the integration range and sum
        # This gives the expected value for each part of the DFL prediction (lt, rb)
        # pred_dist_softmax: (N, 4, dfl_ch)
        # integration_range: (dfl_ch,)
        # Result: (N, 4) -> [pred_l, pred_t, pred_r, pred_b] (offsets from anchor)
        pred_ltrb_offsets = (pred_dist_softmax * integration_range).sum(dim=2) # (N, 4)

        anchor_cx, anchor_cy = anchors_xywh[..., 0], anchors_xywh[..., 1]
        
        # Convert ltrb offsets to xywh boxes
        # pred_l, pred_t are distances from anchor center to left/top
        # pred_r, pred_b are distances from anchor center to right/bottom
        # This assumes the DFL output is predicting offsets for lt,rb from the anchor point.
        # Some YOLO versions predict distance to sides: dl, dt, dr, db
        # If so, then:
        # x1 = anchor_cx - pred_l
        # y1 = anchor_cy - pred_t
        # x2 = anchor_cx + pred_r
        # y2 = anchor_cy + pred_b
        # This needs to be consistent with how GT for DFL is created.
        
        # For now, let's assume the head's DFL output (after self.dfl in eval mode)
        # is already in a form that directly gives dx, dy, dw, dh or similar.
        # The training output of the head is raw distributions *before* the DFL module.
        # The DFL loss needs to work on these raw distributions.
        
        # Let's assume the 4*dfl_ch output corresponds to [dist_l, dist_t, dist_r, dist_b]
        # And the DFL loss expects to regress these distances from anchor points.
        # The `decode_dfl_bboxes` here is for inference/postprocessing or for IoU calculation with GT.
        # For loss, we operate on the distributions directly.

        # This decode is more for when we have applied DFL like in model.eval()
        # For training loss, the target for DFL will be distances, and loss applied to raw distributions.
        # So, this function might be used to get predicted boxes for IoU calculation against GT boxes.
        
        # Let's use the logic from YOLOv8 head's eval path for guidance:
        # a, b = self.dfl(box).chunk(2, 1)  # a=b=[bs,2*self.ch,sum_i(h[i]w[i])]
        # a = anchors.unsqueeze(0) - a
        # b = anchors.unsqueeze(0) + b
        # box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)
        # This implies 'a' and 'b' are offsets for x1y1 and x2y2 from anchor.
        # And 'box' from head is [dist_x1, dist_y1, dist_x2, dist_y2] each of self.ch channels.
        # So pred_dist_softmax would be [softmax(dist_x1), softmax(dist_y1), softmax(dist_x2), softmax(dist_y2)]
        # And pred_ltrb_offsets would be [pred_x1_offset, pred_y1_offset, pred_x2_offset, pred_y2_offset]
        
        # Let pred_dist_softmax be (N, 4, dfl_ch)
        # pred_coords = (pred_dist_softmax * integration_range).sum(dim=2) # (N,4) -> [dx1, dy1, dx2, dy2] offsets
        
        # If anchors are cx,cy,w,h and pred_coords are offsets for l,t,r,b from anchor center
        # Then:
        pred_x1 = anchor_cx - pred_ltrb_offsets[..., 0]
        pred_y1 = anchor_cy - pred_ltrb_offsets[..., 1]
        pred_x2 = anchor_cx + pred_ltrb_offsets[..., 2]
        pred_y2 = anchor_cy + pred_ltrb_offsets[..., 3]
        
        decoded_boxes_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)
        
        # Convert xyxy to xywh
        cx = (decoded_boxes_xyxy[..., 0] + decoded_boxes_xyxy[..., 2]) / 2
        cy = (decoded_boxes_xyxy[..., 1] + decoded_boxes_xyxy[..., 3]) / 2
        w = decoded_boxes_xyxy[..., 2] - decoded_boxes_xyxy[..., 0]
        h = decoded_boxes_xyxy[..., 3] - decoded_boxes_xyxy[..., 1]
        return torch.stack([cx, cy, w, h], dim=-1)


    def decode_simple_bboxes(self, pred_reg, anchors_xywh):
        """ Decodes boxes if not using DFL (e.g. direct regression of cx,cy,w,h offsets) """
        # Assuming pred_reg is (N, 4) with [dx, dy, dw, dh] relative to anchor
        # and anchors_xywh are [ax, ay, aw, ah]
        # px = dx * aw + ax
        # py = dy * ah + ay
        # pw = exp(dw) * aw
        # ph = exp(dh) * ah
        # This is a common way, but YOLOv8 uses DFL. This is a fallback/example.
        raise NotImplementedError("Simple bbox decoding not primary for YOLOv8 DFL head")


    def default_assigner(self, pred_bboxes_xywh, pred_cls_scores, gt_bboxes_xywh, gt_labels, anchors_xywh, mask_gt=None):
        """
        Simplified default assigner.
        This is a placeholder for a more sophisticated assigner like TaskAlignedAssigner.
        A real assigner would:
        1. Calculate alignment metric (e.g., IoU * cls_score^alpha) between preds and GTs.
        2. Select top-k positive candidates for each GT.
        3. Handle conflicts if one pred is assigned to multiple GTs.
        
        For this simplified version:
        - For each GT, find the anchor/prediction with the highest IoU.
        - Consider anchors within a center distance radius of GT.
        
        Returns:
            target_bboxes (Tensor): (num_total_anchors, 4)
            target_scores (Tensor): (num_total_anchors, num_classes)
            fg_mask (Tensor): (num_total_anchors,) boolean mask for foreground anchors
        """
        num_anchors = pred_bboxes_xywh.shape[0]
        num_gt = gt_bboxes_xywh.shape[0]

        if num_gt == 0: # No ground truths
            target_bboxes = torch.zeros((num_anchors, 4), device=self.device)
            target_scores = torch.zeros((num_anchors, self.num_classes), device=self.device)
            fg_mask = torch.zeros(num_anchors, dtype=torch.bool, device=self.device)
            return target_bboxes, target_scores, fg_mask, torch.zeros(num_anchors, 4, device=self.device) # Added dummy target_ltrb_for_dfl


        # Calculate IoU between all predicted bboxes and GT bboxes
        # pred_bboxes_xywh: (num_anchors, 4)
        # gt_bboxes_xywh: (num_gt, 4)
        ious = bbox_iou(pred_bboxes_xywh.unsqueeze(1), gt_bboxes_xywh.unsqueeze(0), xywh=True, CIoU=False) # (num_anchors, num_gt)

        # For each GT, find the best matching anchor
        # This is a very simple strategy, real ones are more complex (e.g. select top k for each GT)
        gt_best_iou, gt_best_iou_idx = ious.max(dim=0) # (num_gt,) best iou for each gt, and its anchor index

        # For each anchor, find the best matching GT
        anchor_best_iou, anchor_best_iou_idx = ious.max(dim=1) # (num_anchors,) best iou for each anchor, and its gt index

        fg_mask = torch.zeros(num_anchors, dtype=torch.bool, device=self.device)
        
        # Simple: Assign anchors that have IoU > threshold with any GT
        # Or, assign anchors that are the best match for some GT (more robust to start)
        # Let's use a common approach: an anchor is positive if its IoU with a GT is > some threshold (e.g. 0.5)
        # AND it's among the top-k matches for that GT (SimOTA style, simplified)

        # Simplified: assign each GT to its best IoU anchor, if IoU > 0.2 (low threshold for simplicity)
        # And assign anchors that have IoU > 0.5 with any GT
        # This can lead to multiple anchors for one GT, or one anchor for multiple GTs.
        
        # Let's implement a one-to-one assignment for simplicity first, then refine.
        # Assign each GT to the anchor that has the highest IoU with it.
        # This is still not ideal but a starting point.
        
        # A common strategy for anchor-based detectors (not anchor-free like YOLOv8 often is):
        # - anchors with IoU > 0.7 with any GT are positive
        # - anchors with IoU < 0.3 with all GTs are negative
        # - anchors between 0.3 and 0.7 are ignored.
        # YOLOv8 is anchor-free, assignment is based on grid cells.
        # The `anchors_xywh` here are effectively grid cell centers + dimensions (if fixed per scale).
        # Or, if `make_anchors` from head is used, they are just grid cell centers.
        # The DFL predicts offsets from these grid cell centers.

        # Let's use a simplified dynamic top-k assignment (inspired by SimOTA/TAL)
        # For each GT, select 'k' anchors based on a cost (e.g., cls_cost + iou_cost)
        # For now, just IoU based:
        k = 10 # Number of anchors to assign per GT
        candidate_ious = ious.T # (num_gt, num_anchors)
        
        target_bboxes = torch.zeros((num_anchors, 4), device=self.device)
        target_scores = torch.zeros((num_anchors, self.num_classes), device=self.device)
        # For DFL, target is not just the bbox, but the distribution of l,t,r,b distances
        target_ltrb_for_dfl = torch.zeros((num_anchors, 4), device=self.device) # Target distances for DFL loss


        for i in range(num_gt):
            gt_box_i = gt_bboxes_xywh[i]
            gt_label_i = gt_labels[i]
            ious_with_gt_i = candidate_ious[i] # (num_anchors,)

            # Select top k anchors for this GT based on IoU
            # More advanced: use cost = w1 * cls_cost + w2 * iou_cost
            # cls_cost = -log(pred_cls_scores_for_gt_class)
            # For now, simpler: just IoU.
            
            # Consider only anchors within a certain radius of the GT center for assignment
            # This is often done by checking if GT center falls into an anchor's "responsible area"
            # (e.g., anchor center +/- stride/2)
            # Let's skip this for the first pass to simplify, and assign based on top-k IoU globally.

            num_top_k = torch.min(torch.tensor(k), (ious_with_gt_i > 0.1).sum()).item() # At least some IoU
            if num_top_k == 0: continue

            _, top_k_indices = torch.topk(ious_with_gt_i, int(num_top_k))
            
            fg_mask[top_k_indices] = True
            target_bboxes[top_k_indices] = gt_box_i
            target_scores[top_k_indices, gt_label_i.long()] = 1.0 # One-hot encoding for class

            # Create DFL targets for these assigned anchors
            # Target for DFL is typically the distance from anchor point to the sides of the GT box.
            # anchor_points_xy: (num_anchors, 2) - centers of anchors/grid cells
            # gt_box_i_xyxy: (4,) - [x1,y1,x2,y2]
            
            assigned_anchors_xy = anchors_xywh[top_k_indices, :2] # centers (cx, cy)
            gt_box_i_xyxy = torch.cat([gt_box_i[:2] - gt_box_i[2:]/2, gt_box_i[:2] + gt_box_i[2:]/2])

            # dist_l = anchor_cx - gt_x1
            # dist_t = anchor_cy - gt_y1
            # dist_r = gt_x2 - anchor_cx
            # dist_b = gt_y2 - anchor_cy
            # These distances are what the DFL distribution should learn.
            # These distances should be in units of pixels (or normalized by stride later if needed)
            # The DFL head predicts distributions for these distances.
            # The range of DFL is [0, reg_max-1]. So these distances need to be mapped to this range.
            # Typically, these distances are divided by stride. dist_l / stride, etc.
            
            # Let anchors_xywh be grid cell centers (absolute coords)
            # Let gt_bboxes_xywh be absolute coords
            # The DFL head output (raw) is for l,t,r,b distances from anchor point.
            # So, target_ltrb_for_dfl should be [dist_l, dist_t, dist_r, dist_b]
            
            # gt_ltrb: [gt_x_center - gt_width/2, gt_y_center - gt_height/2, 
            #           gt_x_center + gt_width/2, gt_y_center + gt_height/2]
            gt_x1, gt_y1, gt_x2, gt_y2 = gt_box_i_xyxy
            
            # For each assigned anchor, calculate distance to GT box sides
            # assigned_anchors_xy are (cx, cy)
            target_l = assigned_anchors_xy[:, 0] - gt_x1
            target_t = assigned_anchors_xy[:, 1] - gt_y1
            target_r = gt_x2 - assigned_anchors_xy[:, 0]
            target_b = gt_y2 - assigned_anchors_xy[:, 1]
            
            # Clamp to be non-negative as DFL range is [0, reg_max-1]
            # These are distances, so should be positive if anchor is inside or near GT.
            # If anchor is far, these can be negative. DFL expects positive discrete values.
            # This implies the DFL formulation used in YOLOv8 might be slightly different,
            # or the anchor points are chosen such that these are mostly positive.
            # Ultralytics YOLOv5 DFL: predicts distances from cell top-left to box top-left (dx1, dy1)
            # and cell bottom-right to box bottom-right (dx2, dy2).
            # Let's assume the DFL predicts distances from anchor center to box sides (l,t,r,b) for now.
            
            current_target_ltrb = torch.stack([target_l, target_t, target_r, target_b], dim=1)
            target_ltrb_for_dfl[top_k_indices] = current_target_ltrb

        # Handle cases where one anchor is assigned to multiple GTs (take the one with highest IoU)
        # This simplified assigner doesn't explicitly do that, but fg_mask can be true for multiple GTs' top-k.
        # The target_bboxes and target_scores would be overwritten by later GTs if indices overlap.
        # A proper assigner handles this by, e.g., ensuring each anchor is assigned to at most one GT.
        # For now, this is a simplification.

        return target_bboxes, target_scores, fg_mask, target_ltrb_for_dfl


    def forward(self, preds_from_head, targets_collated):
        """
        Args:
            preds_from_head (list of torch.Tensor): Output from model.head in training mode.
                Each tensor is [B, num_outputs_per_anchor, H, W].
                num_outputs_per_anchor = 4*dfl_ch (for bbox distribution) + num_classes.
            targets_collated (torch.Tensor): Ground truth targets from dataloader.
                Shape: [total_num_targets_in_batch, 6]
                Columns: [batch_img_idx, class_label, x_center, y_center, w, h] (normalized 0-1)
        Returns:
            total_loss (torch.Tensor)
            loss_components (dict): Dictionary of individual loss values (box, cls, dfl)
        """
        batch_size = preds_from_head[0].shape[0]
        
        # Initialize losses
        loss_box_total = torch.tensor(0., device=self.device)
        loss_cls_total = torch.tensor(0., device=self.device)
        loss_dfl_total = torch.tensor(0., device=self.device)

        # Prepare predictions and anchors from all feature levels
        # preds_flat: list of [B, H*W, num_outputs_per_anchor]
        # anchors_flat_xywh: list of [H*W, 4] (cx,cy,w,h format, absolute pixel coords)
        # Note: YOLOv8 is anchor-free in the sense of pre-defined anchor box shapes per location.
        # It uses grid points as anchor points. The "anchor_wh" might be implicit or based on stride.
        # The `make_anchors` in the head generates anchor center points (sx, sy) and strides.
        # The DFL then predicts distances from these points.
        
        all_level_preds_reshaped = [] # List of (B, H*W, C_total)
        all_level_anchors_xy_stride_normalized = [] # List of (H*W, 2) anchor centers, normalized by stride
                                                    # Or, (H*W, 4) if we consider implicit anchor box based on stride
        
        # Generate anchors using a similar logic to model_head.make_anchors
        # This ensures consistency. Strides are crucial.
        # The head's make_anchors gives absolute (grid unit) coordinates.
        # We need them scaled by stride to be in pixel units for IoU with GTs.
        
        # For each feature level
        for i, P_i in enumerate(preds_from_head):
            B, C_total, H, W = P_i.shape # C_total = 4*dfl_ch + num_classes
            stride_i = self.strides[i]

            # Reshape predictions: (B, C_total, H, W) -> (B, H*W, C_total)
            P_i_reshaped = P_i.view(B, C_total, H * W).permute(0, 2, 1) # (B, H*W, C_total)
            all_level_preds_reshaped.append(P_i_reshaped)

            # Generate anchor points for this level (grid cell centers)
            # These are in grid units (e.g., 0.5, 1.5, ... up to H-0.5 or W-0.5)
            sy, sx = torch.meshgrid(torch.arange(H, device=self.device, dtype=torch.float32) + 0.5,
                                    torch.arange(W, device=self.device, dtype=torch.float32) + 0.5,
                                    indexing='ij')
            anchor_points_xy_grid_units = torch.stack((sx, sy), dim=-1).view(-1, 2) # (H*W, 2)
            
            # Convert anchor points to absolute pixel coordinates
            # These are the "anchor points" from which DFL predicts distances.
            anchor_points_xy_pixels = anchor_points_xy_grid_units * stride_i
            all_level_anchors_xy_stride_normalized.append(anchor_points_xy_pixels)

        # Concatenate predictions and anchors from all levels
        # preds_concat: (B, total_num_anchors_across_levels, C_total)
        # anchors_concat_xy: (total_num_anchors_across_levels, 2) - absolute pixel coords
        preds_concat = torch.cat(all_level_preds_reshaped, dim=1)
        anchors_concat_xy = torch.cat(all_level_anchors_xy_stride_normalized, dim=0)
        num_total_anchors = anchors_concat_xy.shape[0]

        # Split concatenated predictions into box distribution and class scores
        # pred_dist_flat: (B, total_num_anchors, 4 * dfl_ch)
        # pred_cls_scores_flat: (B, total_num_anchors, num_classes)
        pred_dist_flat, pred_cls_scores_flat = torch.split(preds_concat, [4 * self.dfl_ch, self.num_classes], dim=2)

        # Loop over each image in the batch
        for b_idx in range(batch_size):
            # Get GTs for the current image
            gt_mask_b = targets_collated[:, 0] == b_idx
            gt_labels_b = targets_collated[gt_mask_b, 1]
            gt_bboxes_normalized_b = targets_collated[gt_mask_b, 2:] # (num_gt_in_img, 4) in [cx,cy,w,h] normalized

            if gt_bboxes_normalized_b.shape[0] == 0: # No GTs for this image
                # All anchors are background, calculate only cls loss for background
                # Or, if assigner handles this, it will return empty fg_mask
                # For now, let's assume assigner handles num_gt = 0
                pass


            # Denormalize GT bboxes to pixel coordinates
            # Assuming input image size used for normalization was passed or known.
            # The strides are related to input size, e.g. 640.
            # A common input size is 640x640.
            # If gt_bboxes_normalized_b are [0,1], multiply by img_w, img_h
            # Let's assume input image size is implicitly handled by strides or fixed (e.g. 640)
            # For simplicity, let's assume GTs are already in pixel coords for now,
            # or that the `decode_dfl_bboxes` and `bbox_iou` handle normalization if needed.
            # The `targets_collated` are normalized. We need image size.
            # This should come from config or be fixed. Let's assume 640x640 for now.
            # This is a common point of error if not handled carefully.
            # The dataset provides normalized coords. So, we need image dims.
            # H_img, W_img = 640, 640 # Example, should be from config
            # gt_bboxes_pixels_b = gt_bboxes_normalized_b.clone()
            # gt_bboxes_pixels_b[:, 0] *= W_img
            # gt_bboxes_pixels_b[:, 1] *= H_img
            # gt_bboxes_pixels_b[:, 2] *= W_img
            # gt_bboxes_pixels_b[:, 3] *= H_img
            # For now, let's assume gt_bboxes_pixels_b are provided directly or this scaling is done elsewhere.
            # The dataset provides targets normalized to [0,1]. This MUST be scaled.
            # Use self.img_size_h and self.img_size_w passed during init.
            gt_bboxes_pixels_b = gt_bboxes_normalized_b.clone()
            gt_bboxes_pixels_b[:, 0::2] *= self.img_size_w # x_center, width
            gt_bboxes_pixels_b[:, 1::2] *= self.img_size_h # y_center, height


            # Predictions for the current image
            pred_dist_b = pred_dist_flat[b_idx] # (total_num_anchors, 4 * dfl_ch)
            pred_cls_b = pred_cls_scores_flat[b_idx] # (total_num_anchors, num_classes)

            # Decode predicted bboxes (for IoU calculation in assigner)
            # Anchors are absolute pixel coords (centers). DFL predicts distances from these.
            # `decode_dfl_bboxes` needs anchor_xywh. Here anchors_concat_xy are just (cx,cy).
            # We need to provide a dummy w,h for anchors if decode_dfl_bboxes expects xywh.
            # However, the DFL logic in head's eval mode:
            #   a, b = self.dfl(box_dist_part).chunk(2,1) --> a,b are offsets for tlbr from anchor point
            #   x1y1 = anchor_points - a ;  x2y2 = anchor_points + b
            #   This means the `pred_ltrb_offsets` from `decode_dfl_bboxes` are directly used.
            #   `decode_dfl_bboxes` needs `anchors_xywh` where xy are centers.
            #   Let's make `anchors_for_decode` by adding dummy w,h to `anchors_concat_xy`.
            dummy_wh = torch.ones_like(anchors_concat_xy) # Dummy w,h, not really used if DFL predicts ltrb offsets
            anchors_for_decode_xywh = torch.cat([anchors_concat_xy, dummy_wh], dim=1)
            
            pred_bboxes_decoded_b = self.decode_bboxes_fn(pred_dist_b, anchors_for_decode_xywh) # (total_anchors, 4) [cx,cy,w,h] pixels

            # Assign targets to anchors/predictions
            # target_bboxes_assigned: (total_anchors, 4) [cx,cy,w,h] pixels, for assigned anchors, else 0
            # target_scores_assigned: (total_anchors, num_classes), one-hot for assigned, else 0
            # fg_mask_assigned: (total_anchors,) boolean, True for foreground anchors
            # target_ltrb_for_dfl_assigned: (total_anchors, 4) [l,t,r,b] distances for DFL targets, for assigned, else 0
            target_bboxes_assigned, target_scores_assigned, fg_mask_assigned, target_ltrb_for_dfl_assigned = \
                self.assigner(pred_bboxes_decoded_b, torch.sigmoid(pred_cls_b), # Pass sigmoid scores to assigner
                              gt_bboxes_pixels_b, gt_labels_b, 
                              anchors_for_decode_xywh, # Pass anchors (cx,cy,w,h) to assigner
                              mask_gt=None) # mask_gt could be used if some GTs are to be ignored

            num_fg = fg_mask_assigned.sum()

            if num_fg > 0:
                # --- 1. Classification Loss ---
                # Only for foreground anchors (assigned ones)
                # And also for background anchors (non-assigned ones)
                # Cls loss is typically calculated over all anchors.
                # Target for background is all zeros. Target for foreground is one-hot class.
                loss_cls_b = self.bce_cls(pred_cls_b, target_scores_assigned) # (total_anchors, num_classes)
                # Sum over classes, then mean over anchors (or mean over positive anchors, sum over all)
                # Common practice: mean over all anchors, weighted by pos/neg.
                # Or, for positive anchors, use their target, for negative anchors, target is 0.
                # The `target_scores_assigned` from assigner should be correctly set up for this.
                # (0 for background, one-hot for foreground)
                loss_cls_total += loss_cls_b.mean() # Mean over all anchors and all classes. Or sum then normalize.
                                                  # Let's take sum over classes and mean over anchors.
                                                  # loss_cls_total += loss_cls_b.sum(dim=1).mean() - Needs careful normalization

                # Let's follow common practice: calculate BCE for all, then average.
                # The `target_scores_assigned` is 0 for bg, one-hot for fg.
                # So BCE applies to all.
                # The loss for background anchors will push their scores to 0.
                # The loss for foreground anchors will push their scores to 1 for the correct class.
                # Normalization: sum of loss values / (num_fg * batch_size) or similar.
                # Or, if bce_cls reduction is 'none', then:
                # loss_cls_total += loss_cls_b.sum() / num_fg  -- this only considers positives, which is not typical for CE.
                # Let's use mean reduction for BCE and sum them up.
                # This means normalizer is (total_anchors * num_classes).
                # If reduction='sum', then normalize by num_fg.
                # Let's assume self.bce_cls has reduction='none'.
                # Then we need to sum it up and normalize appropriately.
                # A common way:
                # Positive samples cls loss: self.bce_cls(pred_cls_b[fg_mask_assigned], target_scores_assigned[fg_mask_assigned])
                # Negative samples cls loss: self.bce_cls(pred_cls_b[~fg_mask_assigned], target_scores_assigned[~fg_mask_assigned]) (target is 0)
                # For now, let's use the simpler:
                loss_cls_total += self.bce_cls(pred_cls_b, target_scores_assigned).mean() # Average over all predictions

                # --- 2. Bbox Regression Loss (IoU Loss) ---
                # Only for foreground anchors
                pred_bboxes_fg = pred_bboxes_decoded_b[fg_mask_assigned] # (num_fg, 4)
                target_bboxes_fg = target_bboxes_assigned[fg_mask_assigned] # (num_fg, 4)
                
                iou_val = bbox_iou(pred_bboxes_fg, target_bboxes_fg, xywh=True, CIoU=(self.iou_type == 'ciou'),
                                   GIoU=(self.iou_type == 'giou'), DIoU=(self.iou_type == 'diou'))
                loss_box_b = (1.0 - iou_val).mean() # Mean over foreground anchors
                if torch.isnan(loss_box_b): loss_box_b = torch.tensor(0., device=self.device)
                loss_box_total += loss_box_b


                # --- 3. DFL Loss (Distribution Focal Loss) ---
                # Only for foreground anchors
                pred_dist_fg = pred_dist_b[fg_mask_assigned] # (num_fg, 4 * dfl_ch)
                target_ltrb_fg = target_ltrb_for_dfl_assigned[fg_mask_assigned] # (num_fg, 4) [dist_l,t,r,b] in pixels
                
                # The DFL target needs to be discretized to the range [0, reg_max-1]
                # target_ltrb_fg are distances in pixels. Divide by stride.
                # The stride varies per level. This means DFL loss should be calculated per level,
                # or targets need to be normalized by their respective strides before concatenation.
                # The `target_ltrb_for_dfl_assigned` was calculated using `anchors_concat_xy` which are absolute.
                # We need to associate fg_mask back to levels to get correct strides.
                
                # This is where it gets tricky: `fg_mask_assigned` is for concatenated anchors.
                # We need to map these foreground predictions back to their original feature levels
                # to use the correct stride for normalizing DFL targets.

                # Alternative: DFL loss operates on unnormalized distances if reg_max is large enough.
                # But typically, distances are divided by stride to make them scale-invariant.
                # Let target_ltrb_for_dfl be distances normalized by stride.
                # The `anchors_concat_xy` are absolute. `target_ltrb_for_dfl_assigned` are absolute distances.
                # We need to find the stride for each fg anchor.
                
                # Reconstruct strides for all anchors:
                strides_per_anchor = []
                current_anchor_idx = 0
                for i_level, P_level in enumerate(preds_from_head):
                    _, _, H, W = P_level.shape
                    num_anchors_level = H * W
                    strides_per_anchor.append(
                        torch.full((num_anchors_level,), self.strides[i_level].item(), device=self.device)
                    )
                strides_concat = torch.cat(strides_per_anchor, dim=0) # (total_num_anchors,)
                strides_fg = strides_concat[fg_mask_assigned] # (num_fg,)

                # Normalize target_ltrb_fg by their respective strides
                # target_ltrb_fg: (num_fg, 4)
                # strides_fg.unsqueeze(1): (num_fg, 1)
                target_ltrb_normalized_fg = target_ltrb_fg / strides_fg.unsqueeze(1) # Distances in "stride units"

                # Discretize normalized targets for DFL (integer part and fractional part for soft label)
                # target_left = target_ltrb_normalized_fg.floor().long().clamp(min=0, max=self.reg_max - 1)
                # target_right = (target_ltrb_normalized_fg + 1).floor().long().clamp(min=0, max=self.reg_max - 1) # Should be target_left + 1 clamped
                # weight_left = (target_ltrb_normalized_fg - target_left.float()).abs() # Should be 1 - (target - floor)
                # weight_right = 1.0 - weight_left
                
                # Simpler DFL target: just use the closest integer bin.
                # Or, use the "integral regression" form of DFL if reg_max = dfl_ch.
                # Target for DFL is l,t,r,b distances. Model predicts distribution for each.
                # Loss is CE between predicted distribution and target distribution (one-hot or two-hot).
                
                # Let pred_dist_fg be (num_fg, 4 * dfl_ch). Reshape to (num_fg*4, dfl_ch)
                # Let target_ltrb_normalized_fg be (num_fg, 4). Reshape to (num_fg*4,)
                pred_dist_reshaped = pred_dist_fg.view(-1, self.dfl_ch) # (num_fg * 4, dfl_ch)
                target_dist_values = target_ltrb_normalized_fg.view(-1) # (num_fg * 4,) values like 3.4, 5.2 etc.

                # Create soft labels for DFL (linear interpolation between two closest bins)
                target_idx_left = target_dist_values.floor().long()
                target_idx_right = (target_dist_values + 1.0).floor().long() # Equivalent to ceil if not integer
                                                                          # Or target_idx_left + 1
                
                weight_right = target_dist_values - target_idx_left.float() # Fractional part
                weight_left = 1.0 - weight_right

                # Clamp indices to be within [0, dfl_ch-1] (or reg_max-1)
                target_idx_left = target_idx_left.clamp(min=0, max=self.dfl_ch - 1)
                target_idx_right = target_idx_right.clamp(min=0, max=self.dfl_ch - 1)

                # DFL loss: CE loss for each of the 4 coordinate distributions
                # pred_dist_reshaped: (N*4, dfl_ch) - logit scores for each bin
                # target_idx_left, target_idx_right: (N*4,) - indices of target bins
                # weight_left, weight_right: (N*4,) - weights for these bins
                
                # Use F.cross_entropy. It expects class indices as target.
                # For soft labels, we can calculate it manually:
                # log_softmax_preds = F.log_softmax(pred_dist_reshaped, dim=1)
                # loss_dfl_b = -( (F.one_hot(target_idx_left, num_classes=self.dfl_ch) * weight_left.unsqueeze(1) + \
                #                   F.one_hot(target_idx_right, num_classes=self.dfl_ch) * weight_right.unsqueeze(1)) * \
                #                   log_softmax_preds ).sum(dim=1).mean()
                # This is equivalent to:
                loss_dfl_b = (F.cross_entropy(pred_dist_reshaped, target_idx_left, reduction='none') * weight_left + \
                              F.cross_entropy(pred_dist_reshaped, target_idx_right, reduction='none') * weight_right).mean()

                if torch.isnan(loss_dfl_b): loss_dfl_b = torch.tensor(0., device=self.device)
                loss_dfl_total += loss_dfl_b
            else: # No foreground anchors for this image
                # Only background classification loss if desired, or skip if num_fg is the normalizer
                # If assigner found no GTs, then target_scores_assigned is all zeros.
                # So, cls loss would be calculated for background.
                loss_cls_b = self.bce_cls(pred_cls_b, target_scores_assigned) # target_scores_assigned is all zeros
                loss_cls_total += loss_cls_b.mean()


        # Normalize losses by batch size
        # The individual losses (box_b, cls_b, dfl_b) are already mean-reduced over anchors/predictions.
        # So, we just sum them up. If they were sum-reduced, divide by batch_size * num_fg or similar.
        # Since b_idx loop is used, we are accumulating sums of means. So divide by batch_size at the end.
        
        loss_box_total /= batch_size
        loss_cls_total /= batch_size
        loss_dfl_total /= batch_size
        
        # Weighted sum of losses
        total_loss = (self.lambda_box * loss_box_total +
                      self.lambda_cls * loss_cls_total +
                      self.lambda_dfl * loss_dfl_total)

        loss_items = {
            "loss_box": loss_box_total.detach().item(),
            "loss_cls": loss_cls_total.detach().item(),
            "loss_dfl": loss_dfl_total.detach().item(),
            "total_loss": total_loss.detach().item()
        }
        return total_loss, loss_items


if __name__ == '__main__':
    # --- Example Usage (for testing the loss structure) ---
    print("Testing ComputeLoss structure...")
    
    # Dummy model head (mocking its relevant attributes for loss)
    class MockHead:
        def __init__(self, dfl_ch=16, num_classes=3, strides=[8,16]):
            self.dfl_ch = dfl_ch
            self.num_classes = num_classes
            self.strides = torch.tensor(strides, dtype=torch.float32)
            # self.reg_max = dfl_ch # For DFL, typically reg_max = dfl_ch (number of bins)
            self.no = 4 * dfl_ch + num_classes # Total outputs per anchor/prediction point

        # Mock make_anchors if loss needs it directly (not used in current loss structure)
        # def make_anchors(self, x_list, strides_tensor): return None, None 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Config params for loss
    num_classes_test = 3
    dfl_channels_test = 16 # reg_max will be dfl_channels_test
    strides_test = [8.0, 16.0] # For two feature levels
    img_size_test = (320,320) # Example image size for testing denormalization


    mock_head = MockHead(dfl_ch=dfl_channels_test, num_classes=num_classes_test, strides=strides_test)
    
    compute_loss_fn = ComputeLoss(
        model_head=mock_head,
        num_classes=num_classes_test,
        device=device,
        img_size=img_size_test, # Pass test image size
        strides=strides_test,
        dfl_ch=dfl_channels_test,
        reg_max=dfl_channels_test # Critical: reg_max for DFL range, usually dfl_ch
    )
    print(f"ComputeLoss initialized with img_size={img_size_test}.")

    # Create dummy predictions from head (two feature levels)
    # Level 1: H=20, W=20 (e.g., input 320 / stride 16)
    # Level 0: H=40, W=40 (e.g., input 320 / stride 8)
    batch_s = 2
    h_lvl0, w_lvl0 = 40, 40 
    h_lvl1, w_lvl1 = 20, 20
    
    # Total channels = 4 * dfl_ch + num_classes
    C_total = 4 * dfl_channels_test + num_classes_test 

    preds_lvl0 = torch.randn(batch_s, C_total, h_lvl0, w_lvl0, device=device)
    preds_lvl1 = torch.randn(batch_s, C_total, h_lvl1, w_lvl1, device=device)
    preds_from_model_head = [preds_lvl0, preds_lvl1]

    # Create dummy targets (collated)
    # [batch_img_idx, class_label, x_center_norm, y_center_norm, w_norm, h_norm]
    # Normalized GTs (image size e.g. 320x320 for these strides/dims)
    targets_collated_test = torch.tensor([
        [0, 0, 0.5, 0.5, 0.2, 0.2],  # Img 0, obj 0
        [0, 1, 0.3, 0.3, 0.1, 0.1],  # Img 0, obj 1
        [1, 2, 0.7, 0.7, 0.3, 0.15]  # Img 1, obj 0
    ], device=device, dtype=torch.float32)

    print(f"Dummy preds shapes: {[p.shape for p in preds_from_model_head]}")
    print(f"Dummy targets shape: {targets_collated_test.shape}")

    # Set image size for denormalization (used in loss forward)
    # This should match the strides and feature map sizes.
    # E.g. if smallest stride is 8 and feature map is 40x40, input is 320x320.
    # compute_loss_fn.img_size_for_norm = (320, 320) # Set this if needed internally
    # The img_size is now part of ComputeLoss __init__ and used internally.

    try:
        total_loss_val, loss_items_val = compute_loss_fn(preds_from_model_head, targets_collated_test)
        print(f"Loss calculation successful.")
        print(f"Total Loss: {total_loss_val.item()}")
        print(f"Loss Components: {loss_items_val}")
        
        # Check if gradients can flow
        if total_loss_val.requires_grad:
            print("Loss requires grad: True (Good)")
            # total_loss_val.backward() # This would need model params for full test
            # print("Backward pass simulated.")
        else:
            print("Loss requires grad: False (Problem if in training context)")

    except Exception as e:
        print(f"Error during ComputeLoss test: {e}")
        import traceback
        traceback.print_exc()

    print("\nNote: This is a structural test. Accuracy of loss values depends on correct assigner logic and DFL details.")
    print("The default_assigner is very basic. A proper one (e.g., TaskAlignedAssigner) is needed for good results.")
    print("DFL target creation and loss calculation details are critical.")

# TODO:
# 1.  Refine the `default_assigner`. It's very basic. A proper TaskAlignedAssigner or SimOTA is needed.
#     This involves cost matrix calculation (cls_cost + iou_cost) and dynamic top-k selection.
# 2.  DFL target (`target_ltrb_for_dfl_assigned`): Ensure these distances are correctly calculated
#     (e.g. from anchor point to GT box sides) and handled by the DFL loss part.
#     The current DFL target is absolute distances. These are normalized by stride before DFL loss.
#     The DFL loss itself (CE with soft labels for two bins) is a common way.
# 3.  Image size for denormalization: Currently hardcoded to 640x640 in loss. This should be passed
#     from config or dataset. (This is now done via __init__)
# 4.  Normalization of losses: Ensure individual loss components are correctly normalized
#     (e.g., by number of foreground samples or batch size). Current approach is mean over items.
# 5.  Consistency with Head's DFL: Ensure the way DFL is handled in loss (target creation, decoding for IoU)
#     is consistent with the head's DFL module and its interpretation of the 4*dfl_ch outputs.
#     The head's eval path `self.dfl(box).chunk(2,1)` implies the raw `box` output (4*dfl_ch)
#     is split into two parts, then processed by DFL. This detail needs to align.
#     The current loss assumes raw `box` output is [dist_l, dist_t, dist_r, dist_b] where each is `dfl_ch` channels.
#     This seems consistent with `pred_dist.view(-1, 4, self.dfl_ch)` if original is `(N, 4*dfl_ch)`.

# Key assumption: The head's output `box` part of `torch.cat((box, cls), dim=1)` in training
# is indeed the raw distributions for [left_dist, top_dist, right_dist, bottom_dist] from anchor point,
# each distribution having `dfl_ch` channels/bins.
# The `model_head.dfl` module (DFL class from components.py) is used in eval mode.
# The DFL class's forward is:
#   def forward(self, x):
#       # x.shape = B, self.c1 * 4, an  (self.c1 = reg_max = 16 for DFL channels)
#       x = x.view(x.shape[0], 4, self.c1, -1).transpose(2, 1).softmax(1) # B, self.c1, 4, an
#       return (x * self.project).sum(1) # B, 4, an (project is torch.arange(reg_max))
# This means `x` fed to DFL is (B, 4*reg_max, num_anchors_total).
# And it reshapes to (B, 4, reg_max, num_anchors_total), then softmax over reg_max dim.
# This matches the loss's handling of pred_dist_flat.
# The `self.project` is `torch.arange(reg_max, ...)`
# So the DFL module directly computes the expected values of the distances.

# The loss computes this expected value (for IoU calculation) and also the CE loss on the distributions.
# This seems consistent.
