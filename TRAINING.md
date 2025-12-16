# SAR Airport Detection - Training Pipeline

This document describes the complete training pipeline for SAR airport object detection using ViT-Tiny + RT-DETR.

## Training Stages

### Stage 1: MAE Pretraining (Self-Supervised)

**Purpose**: Learn spatial representations by reconstructing masked patches

**Command**:
```bash
uv run python -m sar.train.train_mae \
    --img_dir Airport_Dataset_v0_images \
    --output_dir checkpoints/mae \
    --img_size 224 \
    --batch_size 32 \
    --num_epochs 300 \
    --lr 1.5e-4 \
    --mask_ratio 0.75
```

**Key Features**:
- Uses all 624 images (train + val + test) for unsupervised pretraining
- Masks 75% of patches randomly
- Reconstructs masked patches using lightweight decoder
- Heavy augmentation (rotation, flip, distortion, blur, noise)
- Saves encoder weights to `checkpoints/mae/encoder_mae_final.pth`

**Expected Training Time**: ~2-4 hours on GPU (depends on hardware)

---

### Stage 2: SimCLR Pretraining (Contrastive Learning)

**Purpose**: Learn discriminative features through contrastive learning

**Command**:
```bash
uv run python -m sar.train.train_simclr \
    --img_dir Airport_Dataset_v0_images \
    --output_dir checkpoints/simclr \
    --encoder_pretrain_path checkpoints/mae/encoder_mae_final.pth \
    --img_size 224 \
    --batch_size 32 \
    --num_epochs 200 \
    --lr 3e-4 \
    --temperature 0.5
```

**Key Features**:
- Loads MAE pretrained encoder (optional but recommended)
- Creates two augmented views of each image
- Maximizes agreement between positive pairs
- NT-Xent contrastive loss
- Saves encoder weights to `checkpoints/simclr/encoder_simclr_final.pth`

---

### Stage 3: RT-DETR Detection Training (Supervised)

**Purpose**: Fine-tune for object detection with bounding box supervision

**Command**:
```bash
uv run python -m sar.train.train_detection \
    --train_img_dir dataset/train/images \
    --train_label_dir dataset/train/labels \
    --val_img_dir dataset/val/images \
    --val_label_dir dataset/val/labels \
    --output_dir checkpoints/detection \
    --encoder_pretrain_path checkpoints/simclr/encoder_simclr_final.pth \
    --img_size 224 \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 1e-4 \
    --num_queries 100 \
    --freeze_encoder_epochs 10
```

**Key Features**:
- Loads SimCLR pretrained encoder
- Optional encoder freezing for first N epochs (stabilizes training)
- Hungarian matching for prediction-target assignment
- Combined loss: classification + L1 bbox + GIoU
- Heavy augmentation for training (bbox-safe transforms)
- Saves best model based on validation loss

---

## Dataset Split

The dataset has been split into train/val/test with images from the same airport kept together:

- **Train**: 432 images from 74 airports (70%)
- **Val**: 120 images from 21 airports (20%)
- **Test**: 72 images from 12 airports (10%)

Split created with:
```bash
uv run python -m main
```

---

## Augmentation Strategy

### Pretraining (MAE/SimCLR)
- Geometric: rotation (±45°), flip, scale (0.8-1.2), shift, affine, distortion
- Intensity: brightness/contrast (±30%), blur, noise
- Coarse dropout (simulates occlusion)
- Very aggressive to learn robust features

### Detection Training
- Geometric: rotation (±30°), flip, scale, shift
- BBox-safe crop (maintains label integrity)
- Intensity: brightness/contrast, blur, noise
- Moderate augmentation to preserve label accuracy

### Validation/Test
- Resize to 224×224 only
- No augmentation (clean evaluation)

---

## Hardware Requirements

**Minimum**:
- GPU: 4GB VRAM (batch_size=8)
- RAM: 16GB
- Storage: 2GB for checkpoints

**Recommended**:
- GPU: 8GB+ VRAM (batch_size=32)
- RAM: 32GB
- Storage: 5GB for all checkpoints and logs

---

## Monitoring Training

TensorBoard logs are saved in `checkpoints/*/logs/`

View training progress:
```bash
tensorboard --logdir checkpoints
```

**MAE Metrics**:
- Reconstruction loss
- Visualization of original vs reconstructed images

**SimCLR Metrics**:
- Contrastive loss
- Visualization of augmented views

**Detection Metrics**:
- Total loss
- Classification loss (cross-entropy)
- Bbox L1 loss
- GIoU loss

---

## Tips for Training

1. **Start with small dataset**: Test pipeline on a few images first
2. **Monitor overfitting**: With only 432 training images, model may overfit
3. **Adjust batch size**: Lower if GPU memory is insufficient
4. **Learning rate**: Reduce if loss oscillates; increase if converges too slowly
5. **Encoder freezing**: Freeze encoder for 5-10 epochs to stabilize DETR head
6. **Data augmentation**: Critical for small datasets - already very aggressive
7. **Early stopping**: Monitor validation loss, stop if it starts increasing

---

## Next Steps

After training:

1. **Evaluation**: Implement mAP calculation on test set
2. **Inference**: Create inference script for new images
3. **Visualization**: Plot predictions on test images
4. **Analysis**: Analyze failure cases and false positives/negatives
5. **Improvements**: Consider data augmentation, longer training, or ensemble methods

---

## File Structure

```
sar/
├── models/
│   ├── vit.py           # ViT-Tiny encoder
│   ├── mae.py           # MAE pretraining
│   ├── simclr.py        # SimCLR pretraining
│   └── rtdetr.py        # RT-DETR detection head
├── data/
│   └── datasets.py      # Dataset loaders
├── augmentation.py      # Augmentation pipelines
└── train/
    ├── train_mae.py     # MAE training script
    ├── train_simclr.py  # SimCLR training script
    └── train_detection.py  # Detection training script
```
