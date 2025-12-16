# SAR Airport Detection

Object detection for airports in SAR (Synthetic Aperture Radar) satellite imagery using Vision Transformers and RT-DETR.

## Overview

This project implements a complete pipeline for detecting airports in SAR images:

1. **Self-supervised pretraining** (MAE + SimCLR) on all 624 images
2. **Supervised fine-tuning** (RT-DETR) on labeled train/val splits
3. **Heavy data augmentation** to handle small dataset size

## Architecture

```
Image -> ViT-Tiny Encoder -> RT-DETR Head -> Bounding Boxes
         (pretrained)        (100 queries)
```

- **ViT-Tiny**: 5M parameters, 12 layers, 192-dim embeddings
- **Pretraining**: MAE (masked autoencoding) + SimCLR (contrastive learning)
- **Detection**: RT-DETR with Hungarian matching

## Dataset

- **Source**: SAR Airport Dataset v0
- **Total**: 624 images with YOLO format annotations
- **Split** (by airport to avoid data leakage):
  - Train: 432 images (74 airports)
  - Val: 120 images (21 airports)
  - Test: 72 images (12 airports)

## Quick Start

### Installation

```bash
# Install dependencies
uv sync

# Install package in editable mode
uv pip install -e .
```

### Dataset Split

```bash
# Split dataset into train/val/test
uv run python -m main
```

This creates `dataset/train/`, `dataset/val/`, and `dataset/test/` directories.

### Training

See [TRAINING.md](TRAINING.md) for detailed training instructions.

**Quick training pipeline**:

```bash
# Stage 1: MAE pretraining (300 epochs)
uv run python -m sar.train.train_mae \
    --img_dir Airport_Dataset_v0_images \
    --output_dir checkpoints/mae \
    --batch_size 32 \
    --num_epochs 300

# Stage 2: SimCLR pretraining (200 epochs)
uv run python -m sar.train.train_simclr \
    --img_dir Airport_Dataset_v0_images \
    --output_dir checkpoints/simclr \
    --encoder_pretrain_path checkpoints/mae/encoder_mae_final.pth \
    --batch_size 32 \
    --num_epochs 200

# Stage 3: Detection training (100 epochs)
uv run python -m sar.train.train_detection \
    --train_img_dir dataset/train/images \
    --train_label_dir dataset/train/labels \
    --val_img_dir dataset/val/images \
    --val_label_dir dataset/val/labels \
    --output_dir checkpoints/detection \
    --encoder_pretrain_path checkpoints/simclr/encoder_simclr_final.pth \
    --batch_size 8 \
    --num_epochs 100
```

### Viewer App

View dataset with annotations:

```bash
streamlit run app.py
```

Navigate through images with buttons/slider to see bounding boxes overlaid on SAR images.


## Key Features

### Self-Supervised Pretraining

- **MAE**: Masks 75% of patches, learns to reconstruct them
- **SimCLR**: Contrastive learning with dual augmented views
- **Benefit**: Learn SAR-specific features without labels

### Heavy Augmentation

Essential for small dataset (432 training images):

- Geometric: rotation, flip, scale, shift, distortion
- Intensity: brightness/contrast, blur, noise
- Coarse dropout (occlusion simulation)

### Airport-Aware Splitting

Images from the same airport stay in the same split to prevent data leakage and ensure generalization to new airports.

## Requirements

- Python 3.13+
- PyTorch 2.9+
- CUDA-capable GPU (4GB+ VRAM recommended)
- 16GB+ RAM

## Monitoring

View training progress with TensorBoard:

```bash
tensorboard --logdir checkpoints
```

## Tips

1. **Small dataset**: 432 images is small for transformers - pretrain and augment heavily
2. **Overfitting risk**: Monitor validation loss carefully
3. **GPU memory**: Lower batch size if OOM
4. **Encoder freezing**: Freeze encoder for first 5-10 epochs to stabilize DETR head
5. **Checkpoints**: Use best validation checkpoint, not final epoch
