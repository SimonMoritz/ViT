"""Heavy augmentation pipeline for SAR images."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_pretrain_augmentation(img_size=224):
    """
    Heavy augmentation for self-supervised pretraining (MAE/SimCLR).

    For SAR images, we use:
    - Geometric transformations (rotation, flip, affine)
    - Intensity transformations (brightness, contrast, blur)
    - Advanced augmentations (coarse dropout, grid distortion)

    Args:
        img_size: Target image size
    Returns:
        Albumentations transform
    """
    return A.Compose([
        # Resize and crop
        A.Resize(int(img_size * 1.1), int(img_size * 1.1)),
        A.RandomCrop(img_size, img_size),

        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=45,
            border_mode=0,
            p=0.7
        ),

        # Affine transformations
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            rotate=(-15, 15),
            shear=(-10, 10),
            p=0.5
        ),

        # Elastic/Grid distortions
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=1.0),
        ], p=0.3),

        # Intensity augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        # Noise
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.3),

        # Coarse dropout (similar to cutout)
        A.CoarseDropout(
            max_holes=8,
            max_height=int(img_size * 0.1),
            max_width=int(img_size * 0.1),
            fill_value=0,
            p=0.3
        ),

        # Normalize (assuming grayscale or RGB with similar statistics)
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])


def get_simclr_augmentation(img_size=224):
    """
    Two-view augmentation for SimCLR.
    Returns a transform that creates two augmented views of the same image.

    Args:
        img_size: Target image size
    Returns:
        Albumentations transform
    """
    # Similar to pretrain augmentation but potentially stronger
    return A.Compose([
        A.Resize(int(img_size * 1.2), int(img_size * 1.2)),
        A.RandomResizedCrop(
            img_size, img_size,
            scale=(0.6, 1.0),
            ratio=(0.75, 1.33),
            p=1.0
        ),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # Color jitter (works for SAR too, treating as intensity variations)
        A.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
            p=0.8
        ),

        # Blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.5),

        # Noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

        # Normalize
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])


def get_detection_train_augmentation(img_size=224):
    """
    Augmentation for detection training (with bounding boxes).

    Args:
        img_size: Target image size
    Returns:
        Albumentations transform with bbox support
    """
    return A.Compose([
        A.Resize(img_size, img_size),

        # Geometric augmentations (bbox-safe)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=30,
            border_mode=0,
            p=0.7
        ),

        # Random crop/scale (bbox-safe)
        A.RandomSizedBBoxSafeCrop(
            height=img_size,
            width=img_size,
            erosion_rate=0.2,
            p=0.5
        ),

        # Intensity augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),

        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),

        # Normalize
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',  # Normalized (cx, cy, w, h)
        label_fields=['class_labels'],
        min_visibility=0.3,  # Keep boxes with at least 30% visible
    ))


def get_detection_val_augmentation(img_size=224):
    """
    Minimal augmentation for validation/testing.

    Args:
        img_size: Target image size
    Returns:
        Albumentations transform
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
    ))


class DualViewTransform:
    """
    Transform that creates two different augmented views of the same image.
    Used for SimCLR training.
    """

    def __init__(self, transform):
        """
        Args:
            transform: Albumentations transform to apply
        """
        self.transform = transform

    def __call__(self, image):
        """
        Args:
            image: numpy array (H, W, C)
        Returns:
            view1, view2: Two augmented views of the image
        """
        view1 = self.transform(image=image)['image']
        view2 = self.transform(image=image)['image']
        return view1, view2


if __name__ == "__main__":
    import cv2
    from pathlib import Path

    # Test augmentations
    img_path = Path("Airport_Dataset_v0_images")
    images = sorted(img_path.glob("*.jpg"))

    if images:
        img = cv2.imread(str(images[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(f"Original image shape: {img.shape}")

        # Test pretrain augmentation
        aug = get_pretrain_augmentation(224)
        augmented = aug(image=img)['image']
        print(f"Augmented shape: {augmented.shape}")

        # Test dual view
        dual = DualViewTransform(get_simclr_augmentation(224))
        view1, view2 = dual(img)
        print(f"View 1 shape: {view1.shape}")
        print(f"View 2 shape: {view2.shape}")

        # Test detection augmentation
        bboxes = [[0.5, 0.5, 0.2, 0.3]]  # cx, cy, w, h normalized
        labels = [0]
        det_aug = get_detection_train_augmentation(224)
        det_result = det_aug(image=img, bboxes=bboxes, class_labels=labels)
        print(f"Detection augmented shape: {det_result['image'].shape}")
        print(f"Augmented bboxes: {det_result['bboxes']}")
    else:
        print("No images found for testing")
