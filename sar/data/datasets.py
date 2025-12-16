"""Dataset classes for SAR airport detection."""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np


class PretrainDataset(Dataset):
    """
    Dataset for self-supervised pretraining (MAE/SimCLR).
    Loads all images without labels.
    """

    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir: Directory containing images (can be str or Path)
            transform: Albumentations transform or callable
        """
        self.img_dir = Path(img_dir)
        self.transform = transform

        # Get all jpg images
        self.image_paths = sorted(self.img_dir.glob("*.jpg"))
        assert len(self.image_paths) > 0, f"No images found in {img_dir}"

        print(f"PretrainDataset: Loaded {len(self.image_paths)} images from {img_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx: Index
        Returns:
            If transform is DualViewTransform: (view1, view2)
            Otherwise: image tensor
        """
        img_path = self.image_paths[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        # Apply transform
        if self.transform is not None:
            # Check if it's a dual view transform (for SimCLR)
            if hasattr(self.transform, '__call__') and hasattr(self.transform, 'transform'):
                # DualViewTransform
                view1, view2 = self.transform(image)
                return view1, view2
            else:
                # Regular transform
                image = self.transform(image=image)['image']

        return image


class DetectionDataset(Dataset):
    """
    Dataset for object detection training.
    Loads images with corresponding bounding box annotations.
    """

    def __init__(self, img_dir, label_dir, transform=None, return_dict=False):
        """
        Args:
            img_dir: Directory containing images
            label_dir: Directory containing YOLO format labels
            transform: Albumentations transform with bbox support
            return_dict: If True, return dict format for DETR loss
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.return_dict = return_dict

        # Get all images
        self.image_paths = sorted(self.img_dir.glob("*.jpg"))
        assert len(self.image_paths) > 0, f"No images found in {img_dir}"

        # Verify corresponding labels exist
        for img_path in self.image_paths:
            label_path = self.label_dir / (img_path.stem + ".txt")
            assert label_path.exists(), f"Missing label for {img_path.name}"

        print(f"DetectionDataset: Loaded {len(self.image_paths)} images from {img_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx: Index
        Returns:
            If return_dict=True:
                image: (C, H, W) tensor
                target: dict with 'labels' and 'boxes'
            Otherwise:
                image: (C, H, W) tensor
                boxes: (N, 4) tensor in YOLO format (cx, cy, w, h)
                labels: (N,) tensor of class labels
        """
        img_path = self.image_paths[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        # Load labels (YOLO format: class cx cy w h, normalized)
        boxes = []
        labels = []

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    boxes.append([cx, cy, w, h])
                    labels.append(cls)

        # Handle empty annotations
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

        # Apply augmentation
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                bboxes=boxes.tolist() if len(boxes) > 0 else [],
                class_labels=labels.tolist() if len(labels) > 0 else []
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32) if transformed['bboxes'] else np.zeros((0, 4), dtype=np.float32)
            labels = np.array(transformed['class_labels'], dtype=np.int64) if transformed['class_labels'] else np.zeros((0,), dtype=np.int64)
        else:
            # Convert to tensor manually if no transform
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Convert to tensors
        boxes = torch.from_numpy(boxes)
        labels = torch.from_numpy(labels)

        if self.return_dict:
            target = {
                'labels': labels,
                'boxes': boxes,
            }
            return image, target
        else:
            return image, boxes, labels


def collate_fn_detection(batch):
    """
    Custom collate function for detection dataset.
    Handles variable number of boxes per image.

    Args:
        batch: List of (image, target_dict) tuples
    Returns:
        images: (B, C, H, W) tensor
        targets: List of dicts, each with 'labels' and 'boxes'
    """
    images = []
    targets = []

    for item in batch:
        if len(item) == 2:
            # Dict format
            image, target = item
            images.append(image)
            targets.append(target)
        else:
            # Tuple format
            image, boxes, labels = item
            images.append(image)
            targets.append({'labels': labels, 'boxes': boxes})

    images = torch.stack(images, dim=0)
    return images, targets


if __name__ == "__main__":
    from sar.augmentation import get_pretrain_augmentation, get_detection_train_augmentation, DualViewTransform, get_simclr_augmentation

    # Test PretrainDataset
    print("Testing PretrainDataset...")
    pretrain_ds = PretrainDataset(
        img_dir="Airport_Dataset_v0_images",
        transform=get_pretrain_augmentation(224)
    )
    print(f"Dataset size: {len(pretrain_ds)}")
    img = pretrain_ds[0]
    print(f"Image shape: {img.shape}")

    # Test SimCLR dual view
    print("\nTesting DualViewTransform...")
    simclr_ds = PretrainDataset(
        img_dir="Airport_Dataset_v0_images",
        transform=DualViewTransform(get_simclr_augmentation(224))
    )
    view1, view2 = simclr_ds[0]
    print(f"View1 shape: {view1.shape}, View2 shape: {view2.shape}")

    # Test DetectionDataset
    print("\nTesting DetectionDataset...")
    det_ds = DetectionDataset(
        img_dir="dataset/train/images",
        label_dir="dataset/train/labels",
        transform=get_detection_train_augmentation(224),
        return_dict=True
    )
    print(f"Dataset size: {len(det_ds)}")
    img, target = det_ds[0]
    print(f"Image shape: {img.shape}")
    print(f"Num boxes: {len(target['boxes'])}")
    print(f"Boxes: {target['boxes']}")
    print(f"Labels: {target['labels']}")

    # Test collate function
    from torch.utils.data import DataLoader
    print("\nTesting DataLoader with collate_fn...")
    loader = DataLoader(det_ds, batch_size=2, collate_fn=collate_fn_detection, shuffle=True)
    images, targets = next(iter(loader))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch targets length: {len(targets)}")
    for i, t in enumerate(targets):
        print(f"  Sample {i}: {len(t['boxes'])} boxes")
