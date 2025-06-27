import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO # Added import for pycocotools

class COCODataset(Dataset):
    def __init__(self, images_dir, annotations_file, transform=None, target_transform=None):
        """
        Args:
            images_dir (string): Directory with all the images.
            annotations_file (string): Path to the COCO annotations json file.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional transform to be applied on the target.
        """
        self.images_dir = images_dir
        self.annotations_file = annotations_file
        self.transform = transform
        self.target_transform = target_transform

        if not os.path.exists(self.annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
        if not os.path.isdir(self.images_dir):
            raise NotADirectoryError(f"Images directory not found: {self.images_dir}")

        self.coco = COCO(annotations_file)
        self.image_ids = list(sorted(self.coco.imgs.keys()))

        # Create a mapping from COCO category IDs to contiguous IDs (0 to num_classes-1)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.label2cat = {i: cat_id for i, cat_id in enumerate(self.cat_ids)}

        print(f"Initialized COCODataset: Found {len(self.image_ids)} images in {annotations_file}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset at the specified index.
        Args:
            idx (int or torch.Tensor): Index of the sample to retrieve. If a torch.Tensor is provided, it will be converted to a list and then to an integer.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - image (torch.Tensor): The image tensor of shape (C, H, W), where C is the number of channels.
                - targets (torch.Tensor): A tensor of shape (N, 5), where N is the number of valid bounding boxes. Each row contains [class_label, x_center, y_center, width, height] with coordinates normalized to [0, 1].
        Raises:
            TypeError: If the provided index is not an integer.
        Notes:
            - If the image file is missing, a dummy image tensor and empty targets tensor are returned.
            - Only non-crowd and valid bounding boxes (area > 0, width > 0, height > 0) are included in targets.
            - If no valid annotations are found, an empty targets tensor of shape (0, 5) is returned.
            - If `self.transform` is provided, it is applied to the image (and possibly targets, depending on implementation).
            - If `self.target_transform` is provided, it is applied to the targets.
            - If no transform is provided, the image is converted to a tensor using torchvision's `ToTensor`.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not isinstance(idx, int):
            raise TypeError(f"Index should be an integer, got {type(idx)} instead.")

        img_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotations = self.coco.loadAnns(ann_ids)

        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image file not found {img_path} for image ID {img_id}. Skipping.")
            # This should ideally be handled more gracefully, perhaps by returning None
            # or by pre-filtering missing images during initialization.
            # For now, let's return a dummy tensor and empty targets if an image is missing.
            # A better approach would be to filter self.image_ids during __init__
            # Or, if this happens rarely, the dataloader's collate_fn might need to handle None returns.
            return torch.zeros(3, 640, 640), torch.empty(0, 5) # Dummy data

        # Prepare targets: [class_label, x_center, y_center, width, height] (normalized)
        targets = []
        for ann in coco_annotations:
            if ann.get('iscrowd', 0) == 0 and ann['area'] > 0: # Filter out crowd annotations and empty boxes
                # COCO bbox format: [x_min, y_min, width, height]
                bbox = ann['bbox']
                category_id = ann['category_id']
                class_label = self.cat2label.get(category_id)

                if class_label is None:
                    # This can happen if the annotation file has categories not in self.cat_ids
                    # or if self.cat_ids was filtered in a specific way.
                    # print(f"Warning: Category ID {category_id} not in cat2label mapping. Skipping annotation.")
                    continue

                x_min, y_min, w, h = bbox
                
                # Image dimensions
                img_w, img_h = image.size

                if w <= 0 or h <= 0: # Skip invalid bounding boxes
                    continue

                # Convert to [x_center, y_center, width, height] normalized
                x_center = (x_min + w / 2) / img_w
                y_center = (y_min + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                targets.append([class_label, x_center, y_center, norm_w, norm_h])

        targets = torch.tensor(targets, dtype=torch.float32)
        if targets.shape[0] == 0: # Handle cases with no valid annotations
            targets = torch.empty(0, 5)


        sample = {'image': image, 'targets': targets, 'img_id': img_id, 'img_path': img_path}

        if self.transform:
            # The transform should handle both image and target transformations if targets need to be adjusted
            # e.g., after resizing or flipping the image.
            # For simplicity, a common approach is that `transform` only modifies the image,
            # and target adjustments (if any due to image scaling) are handled here or by collate_fn.
            # However, augmentations like flips require target updates.
            # A more robust transform pipeline (e.g. using albumentations) would handle this.
            # For now, let's assume self.transform primarily handles image tensor conversion and normalization.
            # And any geometric transforms that affect bboxes should also update targets.
            
            # A typical simple transform:
            # sample['image'] = transforms.ToTensor()(sample['image'])
            # sample['image'] = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(sample['image'])
            
            # For now, we pass both image and targets to transform, assuming it knows how to handle them.
            # This is a common pattern if using libraries like albumentations.
            # If self.transform is just a torchvision.transforms.Compose for images,
            # then targets should not be passed or handled separately.
            
            # Let's assume a simple transform for the image for now.
            # Target transformation logic will be added when we define specific augmentations.
            transformed_image = self.transform(sample['image'])
            sample['image'] = transformed_image
            # Note: If self.transform resizes the image, the normalized bbox coordinates remain valid.
            # If it crops or pads in a way that changes the relative positions, targets need updates.
            
        # Target transform could be used for formatting targets for the model, e.g., padding.
        if self.target_transform:
            sample['targets'] = self.target_transform(sample['targets'])
            
        # If no transform is provided, ensure image is a tensor for the collate_fn
        if not isinstance(sample['image'], torch.Tensor):
            # Basic conversion to tensor if no other transform did it
            import torchvision.transforms as T
            sample['image'] = T.ToTensor()(sample['image'])

        return sample['image'], sample['targets']

    def collate_fn(self, batch):
        """
        Custom collate_fn for dealing with varying numbers of target boxes per image.
        Pads targets to the max number of objects in a batch and adds a batch index.
        Args:
            batch: list of tuples (image, targets)
        Returns:
            images: tensor of shape (batch_size, C, H, W)
            targets: tensor of shape (total_num_targets_in_batch, 6)
                     where each row is [batch_img_idx, class_label, x_center, y_center, w, h]
        """
        images = []
        targets_list = []

        for i, (img, tgts) in enumerate(batch):
            images.append(img) # Assuming img is already a tensor
            if tgts.shape[0] > 0:
                # Add batch index to targets
                batch_idx_column = torch.full((tgts.shape[0], 1), i, dtype=tgts.dtype, device=tgts.device)
                targets_list.append(torch.cat([batch_idx_column, tgts], dim=1))
        
        if not images: # Should not happen if dataset is not empty and no errors during __getitem__
             return torch.empty(0,3,640,640), torch.empty(0,6)


        # Stack images. This assumes all images have been transformed to the same size.
        images_tensor = torch.stack(images, 0)

        if not targets_list: # If no targets in the entire batch
            return images_tensor, torch.empty(0, 6)
            
        targets_tensor = torch.cat(targets_list, 0)
        return images_tensor, targets_tensor


if __name__ == '__main__':
    # Example Usage (requires COCO dataset to be set up)
    # Replace with actual paths to your COCO dataset
    TRAIN_IMG_DIR = "/coco/train2017"
    TRAIN_ANN_FILE = "/coco/annotations/instances_train2017.json"
    
    VAL_IMG_DIR = "/coco/val2017"
    VAL_ANN_FILE = "/coco/annotations/instances_val2017.json"

    # Basic transform for testing (resize and convert to tensor)
    import torchvision.transforms as T
    img_size = 640
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Test with dummy paths if real COCO is not available, expecting errors
    # Create dummy files for testing structure if needed
    dummy_ann_content = {
        "images": [{"id": 1, "file_name": "dummy.jpg", "width": 640, "height": 480}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 1, 
            "bbox": [10, 10, 50, 50], "area": 2500, "iscrowd": 0
        }],
        "categories": [{"id": 1, "name": "object"}]
    }
    
    # Create dummy structure for testing if COCO paths are not set
    # This part is for local testing of the script and might not run in the agent's environment
    # without actual file creation permissions or pre-existing dummy files.
    
    use_dummy_data = False
    if not (os.path.exists(TRAIN_IMG_DIR) and os.path.exists(TRAIN_ANN_FILE)):
        print("COCO paths not found, attempting to set up dummy data for testing structure.")
        use_dummy_data = True
        
        dummy_root = "dummy_coco_data"
        TRAIN_IMG_DIR = os.path.join(dummy_root, "images")
        TRAIN_ANN_FILE = os.path.join(dummy_root, "annotations.json")
        
        os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
        with open(TRAIN_ANN_FILE, 'w') as f:
            json.dump(dummy_ann_content, f)
        
        # Create a dummy image
        try:
            from PIL import Image
            dummy_image = Image.new('RGB', (640, 480), color = 'red')
            dummy_image.save(os.path.join(TRAIN_IMG_DIR, "dummy.jpg"))
            print(f"Created dummy image at {os.path.join(TRAIN_IMG_DIR, 'dummy.jpg')}")
        except ImportError:
            print("Pillow not installed, cannot create dummy image.")
        print(f"Dummy annotation at {TRAIN_ANN_FILE}")


    print(f"Attempting to load dataset from: {TRAIN_ANN_FILE} and {TRAIN_IMG_DIR}")
    
    try:
        coco_train_dataset = COCODataset(
            images_dir=TRAIN_IMG_DIR,
            annotations_file=TRAIN_ANN_FILE,
            transform=transform
        )

        if len(coco_train_dataset) > 0:
            print(f"Successfully loaded {len(coco_train_dataset)} samples from the dataset.")
            
            # Test __getitem__
            img, targets = coco_train_dataset[0]
            print("Sample image shape:", img.shape) # Expected: [C, H, W]
            print("Sample targets shape:", targets.shape) # Expected: [num_obj, 5]
            print("Sample targets:", targets)

            # Test DataLoader
            from torch.utils.data import DataLoader
            train_dataloader = DataLoader(
                coco_train_dataset,
                batch_size=4,
                shuffle=True,
                num_workers=0, # Set to 0 for easier debugging, use >0 for performance
                collate_fn=coco_train_dataset.collate_fn 
            )

            # Iterate over a few batches
            for i, (batch_images, batch_targets) in enumerate(train_dataloader):
                print(f"\nBatch {i+1}")
                print("Batch images shape:", batch_images.shape) # Expected: [batch_size, C, H, W]
                print("Batch targets shape:", batch_targets.shape) # Expected: [total_num_obj_in_batch, 6]
                # print("Batch targets:", batch_targets)
                if i >= 1: # Print 2 batches
                    break
            
            print("\nCOCODataset structure seems OK.")
        else:
            print("Dataset loaded but is empty. Check paths and annotation file content.")
            if use_dummy_data:
                 print("Note: Using dummy data. Ensure dummy files were created correctly if this fails.")

    except Exception as e:
        print(f"Error during COCODataset test: {e}")
        print("Please ensure COCO dataset paths are correctly set or dummy data is properly created.")
        if "pycocotools" in str(e).lower():
            print("This might be due to 'pycocotools' not being installed. Try: pip install pycocotools")

    # Clean up dummy data if created
    if use_dummy_data and os.path.exists("dummy_coco_data"):
        import shutil
        # shutil.rmtree("dummy_coco_data")
        # print("Cleaned up dummy_coco_data directory.")
        # For the agent, let's not remove it, as it might be useful for subsequent steps or tests.
        print("Dummy data was used. If you have real COCO paths, please update them in the script for actual testing.")

