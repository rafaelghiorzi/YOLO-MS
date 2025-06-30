import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np # For albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 # For PadIfNeeded border_mode, though Resize is used now.

class COCODataset(Dataset):
    def __init__(self, images_dir, annotations_file, transform_params=None, is_train=True, 
                 img_size=(640,640), num_classes=80):
        """
        Args:
            images_dir (string): Directory with all the images.
            annotations_file (string): Path to the COCO annotations json file.
            transform_params (dict, optional): Dictionary of augmentation parameters from config.
            is_train (bool): If True, applies training augmentations. Otherwise, basic validation transform.
            img_size (tuple): Target image size (height, width) for resizing.
            num_classes (int): Number of classes in the dataset. Used for sanity checking cat_ids.
        """
        self.images_dir = images_dir
        self.annotations_file = annotations_file
        self.is_train = is_train
        self.img_h, self.img_w = img_size
        self.num_classes = num_classes

        if transform_params is None:
            transform_params = {}
        self.transform_params = transform_params

        if not os.path.exists(self.annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
        if not os.path.isdir(self.images_dir):
            raise NotADirectoryError(f"Images directory not found: {self.images_dir}")

        self.coco = COCO(annotations_file)
        self.image_ids = list(sorted(self.coco.imgs.keys()))
        
        # Filter out image_ids for which images are missing
        self._filter_missing_images()

        self.cat_ids = self.coco.getCatIds()
        # This mapping logic might need to be more robust if dataset categories don't align with num_classes easily.
        # For instance, if config provides a specific list of category names to use.
        # For now, if num_classes from config is less than actual categories, we take the first N.
        # This assumes that the category IDs in the annotation file are somewhat ordered or that
        # using the first `num_classes` found by `getCatIds()` is acceptable.
        # A better approach for subsets would be to map specific category names from config to their IDs.
        if self.num_classes < len(self.cat_ids):
            print(f"Warning: Dataset has {len(self.cat_ids)} categories, but model is configured for {self.num_classes}. "
                  f"Using the first {self.num_classes} categories found by COCO API. "
                  "Ensure this aligns with the intended subset of classes.")
            self.cat_ids = self.cat_ids[:self.num_classes]
        elif self.num_classes > len(self.cat_ids):
             print(f"Warning: Model configured for {self.num_classes} classes, but dataset only has {len(self.cat_ids)}. "
                   "This might lead to issues if model expects more classes than available.")


        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.label2cat = {i: cat_id for i, cat_id in enumerate(self.cat_ids)}
        
        self.transform = self._setup_transform()

        print(f"Initialized COCODataset: Found {len(self.image_ids)} images in {annotations_file}. Using {'training' if is_train else 'validation'} transforms.")

    def _filter_missing_images(self):
        valid_image_ids = []
        for img_id in self.image_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.images_dir, img_info['file_name'])
            if os.path.exists(img_path):
                valid_image_ids.append(img_id)
            else:
                print(f"Warning: Image file not found {img_path} for image ID {img_id}. Filtering out this image.")
        original_count = len(self.image_ids)
        self.image_ids = valid_image_ids
        if len(self.image_ids) < original_count:
            print(f"Filtered out {original_count - len(self.image_ids)} missing images. Remaining images: {len(self.image_ids)}")

    def _setup_transform(self):
        bbox_params = A.BboxParams(format='coco', 
                                   label_fields=['class_labels'], 
                                   min_visibility=0.1, 
                                   min_area=1) # Min area in pixels for a bbox to be kept after transform

        transform_list = []
        if self.is_train:
            # HSV Augmentation
            if self.transform_params.get('hsv_h', 0) > 0 or \
               self.transform_params.get('hsv_s', 0) > 0 or \
               self.transform_params.get('hsv_v', 0) > 0:
                transform_list.append(A.HueSaturationValue(
                    hue_shift_limit=int(self.transform_params.get('hsv_h', 0) * 100), # Albumentations HSV is typically [-x, x]
                    sat_shift_limit=int(self.transform_params.get('hsv_s', 0) * 100),
                    val_shift_limit=int(self.transform_params.get('hsv_v', 0) * 100),
                    p=0.5 
                ))
            # Geometric Augmentations
            if self.transform_params.get('degrees', 0) > 0:
                transform_list.append(A.Rotate(limit=self.transform_params.get('degrees', 0), p=0.5))
            if self.transform_params.get('translate', 0) > 0:
                # Albumentations translate is fraction of image size
                transform_list.append(A.ShiftScaleRotate(shift_limit=self.transform_params.get('translate', 0), scale_limit=0, rotate_limit=0, p=0.5))
            
            # RandomScale needs careful parameterization. `scale` in Ultralytics is gain.
            # Albumentations RandomScale scale_limit is (lower_bound - 1.0, upper_bound - 1.0)
            # If config 'scale' is 0.5, it means image scaled from 0.5 to 1.5 of original.
            # So scale_limit would be (-0.5, 0.5) for A.RandomScale if interpreting 'scale' as max deviation from 1.0
            # Or, if 'scale' from config is a single value like 0.5 (gain), then scale_limit could be e.g. [1-scale, 1+scale]
            # For Ultralytics, scale=0.5 means image_scale *= random.uniform(1 - hyp['scale'], 1 + hyp['scale'])
            # Let's assume config 'scale' = 0.5 means range [0.5, 1.5].
            # A.RandomScale(scale_limit=0.5) means scale between 0.5 and 1.5. This matches.
            if self.transform_params.get('scale', 0) > 0:
                 transform_list.append(A.RandomScale(scale_limit=self.transform_params.get('scale', 0), p=0.5))
            
            # Corrected shear implementation using Affine
            if self.transform_params.get('shear', 0) > 0:
                shear_val = self.transform_params.get('shear', 0)
                transform_list.append(A.Affine(shear={'x': (-shear_val, shear_val), 'y': (-shear_val, shear_val)}, p=0.5))
            
            if self.transform_params.get('perspective', 0) > 0:
                 transform_list.append(A.Perspective(scale=(0.0, self.transform_params.get('perspective',0.05)), p=0.5)) # perspective scale is max_coef

            if self.transform_params.get('fliplr', 0) > 0:
                transform_list.append(A.HorizontalFlip(p=self.transform_params.get('fliplr', 0)))
            if self.transform_params.get('flipud', 0) > 0:
                transform_list.append(A.VerticalFlip(p=self.transform_params.get('flipud', 0)))

        # Common transforms (Resize, Normalize, ToTensor)
        # Using direct Resize. For letterboxing/padding, PadIfNeeded would be used.
        transform_list.append(A.Resize(height=self.img_h, width=self.img_w, interpolation=cv2.INTER_LINEAR))
        transform_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        transform_list.append(ToTensorV2())
        
        return A.Compose(transform_list, bbox_params=bbox_params)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        if not isinstance(idx, int):
            raise TypeError(f"Index should be an integer, got {type(idx).__name__} instead.")

        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, img_info['file_name'])

        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except FileNotFoundError: # Should ideally not happen if _filter_missing_images worked
            print(f"Critical Error: Image {img_path} (ID: {img_id}) not found despite pre-filtering. Returning dummy data.")
            return torch.zeros(3, self.img_h, self.img_w), torch.empty(0, 5)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotations = self.coco.loadAnns(ann_ids)
        
        bboxes_coco_fmt = [] # List of [x_min, y_min, width, height] in pixels
        class_labels = []    # List of class labels

        for ann in coco_annotations:
            if ann.get('iscrowd', 0) == 0 and ann['area'] > 0:
                coco_bbox = ann['bbox'] 
                category_id = ann['category_id']
                class_label = self.cat2label.get(category_id)

                if class_label is None: continue # Skip if class not in our defined map
                
                x_min, y_min, w, h = coco_bbox
                if w <= 0 or h <= 0: continue

                bboxes_coco_fmt.append([x_min, y_min, w, h])
                class_labels.append(class_label)
        
        transformed_data = None
        try:
            transformed_data = self.transform(image=image, bboxes=bboxes_coco_fmt, class_labels=class_labels)
            image_transformed = transformed_data['image']
            bboxes_transformed = transformed_data['bboxes'] 
            final_class_labels = transformed_data['class_labels'] 
        except Exception as e:
            print(f"Error during albumentations transform for image ID {img_id} ({img_path}): {e}")
            print(f"Original bboxes: {bboxes_coco_fmt}, Original class_labels: {class_labels}")
            # Fallback: return original image (resized) and original GTs (normalized) without augmentation
            # This is better than crashing or returning dummy data if only some augmentations fail.
            # However, if Resize itself fails, this won't help.
            try:
                fallback_transform = A.Compose([
                    A.Resize(height=self.img_h, width=self.img_w),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
                
                transformed_data = fallback_transform(image=np.array(Image.open(img_path).convert('RGB')), # Re-load original
                                                      bboxes=bboxes_coco_fmt, 
                                                      class_labels=class_labels)
                image_transformed = transformed_data['image']
                bboxes_transformed = transformed_data['bboxes']
                final_class_labels = transformed_data['class_labels']
                print("Applied fallback transform (resize, normalize, to_tensor) due to augmentation error.")
            except Exception as fallback_e:
                print(f"Fallback transform also failed: {fallback_e}. Returning dummy data.")
                return torch.zeros(3, self.img_h, self.img_w), torch.empty(0, 5)


        final_targets = []
        if bboxes_transformed: 
            for i, bbox_pixel in enumerate(bboxes_transformed):
                if i >= len(final_class_labels): # Should not happen if albumentations is consistent
                    print(f"Warning: Mismatch between bboxes and class_labels after transform for {img_id}")
                    break 
                class_label = final_class_labels[i] 
                x_min, y_min, w, h = bbox_pixel

                x_center_norm = (x_min + w / 2) / self.img_w
                y_center_norm = (y_min + h / 2) / self.img_h
                w_norm = w / self.img_w
                h_norm = h / self.img_h
                
                if w_norm > 1e-3 and h_norm > 1e-3 and \
                   0 <= x_center_norm <= 1 and 0 <= y_center_norm <= 1 and \
                   0 <= w_norm <=1 and 0 <= h_norm <= 1: # Added tolerance for w,h > 0
                    final_targets.append([class_label, x_center_norm, y_center_norm, w_norm, h_norm])
        
        targets_tensor = torch.tensor(final_targets, dtype=torch.float32)
        if targets_tensor.shape[0] == 0:
            targets_tensor = torch.empty(0, 5)
            
        return image_transformed, targets_tensor

    def collate_fn(self, batch):
        images = []
        targets_list = []
        valid_batch_items = 0 # Count items that are not dummy error items

        for item in batch:
            if item is None: # Should not happen with current error handling, but as safeguard
                continue 
            img, tgts = item
            # Check if it's dummy data (e.g. torch.zeros(3, H, W) and torch.empty(0,5))
            # This check might need to be more robust if dummy data changes
            is_dummy = (img.shape == torch.Size([3, self.img_h, self.img_w]) and torch.all(img == 0) and tgts.shape == torch.Size([0,5]))
            if is_dummy and tgts.numel() == 0 : # A more specific check for our dummy data
                 print("Warning: collate_fn received dummy data for a sample. Skipping it.")
                 continue

            images.append(img) 
            if tgts.shape[0] > 0:
                batch_idx_column = torch.full((tgts.shape[0], 1), valid_batch_items, dtype=tgts.dtype, device=tgts.device) # Use valid_batch_items as batch index
                targets_list.append(torch.cat([batch_idx_column, tgts], dim=1))
            valid_batch_items +=1
        
        if not images: 
             # This means the entire batch consisted of dummy/error data
             print("Warning: collate_fn received an empty batch or all items were errors.")
             return torch.empty(0, 3, self.img_h, self.img_w), torch.empty(0, 6)

        images_tensor = torch.stack(images, 0)
        if not targets_list: 
            return images_tensor, torch.empty(0, 6)
            
        targets_tensor = torch.cat(targets_list, 0)
        return images_tensor, targets_tensor
