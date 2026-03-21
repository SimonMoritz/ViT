# CLAUDE.md — AI Assistant Guide for SAR Airport Detection

This document provides context for AI assistants (Claude and others) working in this repository.

## Project Overview

This is a **Vision Transformer (ViT) based object detection system** for detecting airports in Synthetic Aperture Radar (SAR) satellite imagery. It implements a three-stage training pipeline:

1. **Self-supervised pretraining** — MAE (Masked Autoencoder) + SimCLR (contrastive learning)
2. **Fine-tuning** — RT-DETR (Real-Time Detection Transformer) detection head
3. **Visualization** — Streamlit app for browsing annotated images

**Dataset**: 624 SAR images, 107 airports, YOLO-format bounding box labels.

---

## Repository Structure

```
ViT/
├── main.py                    # Entry point: run dataset splitting
├── app.py                     # Streamlit annotation viewer
├── pyproject.toml             # Project metadata and dependencies
├── uv.lock                    # Locked dependency versions (do not edit manually)
├── .python-version            # Python 3.13
├── README.md                  # Quick start guide
├── TRAINING.md                # Detailed training documentation
└── sar/                       # Main Python package
    ├── __init__.py            # Exports: split_dataset
    ├── dataset.py             # split_dataset() — airport-aware train/val/test split
    ├── augmentation.py        # Albumentations pipelines for all training stages
    ├── models/
    │   ├── vit.py             # ViT-Tiny backbone (192 dim, 12 layers, 3 heads, 5M params)
    │   ├── mae.py             # Masked Autoencoder (75% masking, lightweight decoder)
    │   ├── simclr.py          # SimCLR with NT-Xent loss and projection head
    │   └── rtdetr.py          # RT-DETR detection head + Hungarian matching loss
    ├── data/
    │   └── datasets.py        # PretrainDataset, DetectionDataset, collate_fn_detection
    └── train/
        ├── train_mae.py       # MAE pretraining (300 epochs, batch=32, lr=1.5e-4)
        ├── train_simclr.py    # SimCLR pretraining (200 epochs, batch=32, lr=3e-4)
        └── train_detection.py # Detection fine-tuning (100 epochs, batch=8, lr=1e-4)
```

---

## Development Environment

### Package Manager

This project uses **`uv`** (not pip or conda) for dependency management.

```bash
# Install all dependencies
uv sync

# Add a new dependency
uv add <package>

# Run a script inside the venv
uv run python main.py

# Install package in editable mode
uv pip install -e .
```

### Python Version

Python **3.13** is required (enforced via `.python-version`).

### Key Dependencies

| Package | Purpose |
|---|---|
| torch 2.9.1+ | Model training (CUDA) |
| torchvision 0.24.1 | Image transforms |
| albumentations 2.0.8+ | Augmentation pipelines |
| streamlit 1.52.1 | Annotation viewer |
| tensorboard 2.20.0 | Training monitoring |
| ruff | Linting |
| ty | Type checking (replaces mypy) |

---

## Workflow

### 1. Dataset Preparation

Place the raw SAR dataset in the project root:
```
Airport_Dataset_v0_images/   ← raw images
Airport_Dataset_v0_labels/   ← YOLO-format .txt label files
```

Then split into train/val/test:
```bash
uv run python main.py
```

Output is written to `dataset/train/`, `dataset/val/`, `dataset/test/` — each containing `images/` and `labels/` subdirectories.

**Important**: The split keeps all images of the same airport in the same split to prevent data leakage. Default ratios: 70% train / 20% val / 10% test.

### 2. Pretraining (Recommended)

```bash
# Stage 1: MAE pretraining
uv run python -m sar.train.train_mae

# Stage 2: SimCLR (optionally loads MAE encoder)
uv run python -m sar.train.train_simclr
```

Checkpoints are saved to:
- `checkpoints/mae/encoder_mae_final.pth`
- `checkpoints/simclr/encoder_simclr_final.pth`

### 3. Detection Training

```bash
uv run python -m sar.train.train_detection
```

- Loads pretrained encoder weights
- Freezes encoder for first N epochs (stable fine-tuning)
- Saves best model (by validation loss) to `checkpoints/detection/`

### 4. Dataset Visualization

```bash
uv run streamlit run app.py
```

Opens a browser UI to browse SAR images with bounding box overlays.

### 5. Monitor Training

```bash
tensorboard --logdir runs/
```

---

## Architecture Details

### ViT-Tiny Backbone (`sar/models/vit.py`)

| Parameter | Value |
|---|---|
| Patch size | 16×16 |
| Embedding dim | 192 |
| Depth | 12 transformer blocks |
| Attention heads | 3 |
| MLP ratio | 4× |
| Parameters | ~5M |

- Pre-norm architecture (LayerNorm before attention/MLP)
- Optional CLS token
- Learnable 1D positional embeddings

### MAE (`sar/models/mae.py`)

- Masks **75%** of patches randomly at pretraining
- Lightweight decoder: 4 blocks, 128 dim
- Loss: MSE on masked patches only
- Outputs pretrained encoder weights for downstream tasks

### SimCLR (`sar/models/simclr.py`)

- Projection head: 192 → 512 → 128 dims
- NT-Xent loss at temperature=0.5
- Global average pooling over all patch tokens
- Two augmented views per image via `DualViewTransform`

### RT-DETR (`sar/models/rtdetr.py`)

- **100 learnable object queries**
- **6 decoder layers** (self-attention + cross-attention with encoder)
- **2 output classes**: airport (0) + background (1)
- **Loss components**:
  - Classification (cross-entropy)
  - Bounding box L1 (weight: 5×)
  - GIoU (weight: 2×)
- Uses **Hungarian matching** to assign predictions to ground-truth targets

---

## Data Conventions

### Bounding Box Format

All labels use **YOLO format** (one object per line in `.txt` files):
```
class_id cx cy w h
```
- All values are normalized to `[0, 1]` relative to image dimensions
- `cx, cy` = center coordinates; `w, h` = width/height
- Class IDs: `0` = airport

### Image Format

- **Input resolution**: 224×224 (standard) — configurable per training stage
- **Color space**: RGB (even though SAR images are greyscale, they are typically loaded as 3-channel)
- **Normalization**: ImageNet mean/std (`[0.485, 0.456, 0.406]`, `[0.229, 0.224, 0.225]`)

### Augmentation Strategy

| Stage | Pipeline | Key transforms |
|---|---|---|
| MAE pretraining | `get_pretrain_augmentation()` | Rotation ±45°, flip, scale 0.8–1.2, blur, noise |
| SimCLR pretraining | `get_simclr_augmentation()` | Two-view: color jitter, blur, noise |
| Detection training | `get_detection_train_augmentation()` | Bbox-safe moderate augmentation |
| Detection val/test | `get_detection_val_augmentation()` | Resize only |

---

## Code Conventions

### Style

- **Linting**: ruff (`uv run ruff check .` / `uv run ruff format .`)
- **Type checking**: ty (`uv run ty check sar/`)
- **Docstrings**: Present on all public functions (Args / Returns format)
- **Type hints**: Used throughout, especially in function signatures

### Linting & Type Checking

This project uses **ruff** for linting/formatting and **ty** (by Astral) for type checking. Both are configured in `pyproject.toml`.

#### ruff

```bash
# Check for lint errors
uv run ruff check .

# Auto-fix fixable lint errors
uv run ruff check . --fix

# Format code
uv run ruff format .

# Check formatting without writing
uv run ruff format . --check
```

Configuration lives under `[tool.ruff]` and `[tool.ruff.lint]` in `pyproject.toml`:
- Line length: 100
- Target: Python 3.13
- Enabled rule sets: `E`, `F`, `W` (pycodestyle/pyflakes errors), `I` (isort), `UP` (pyupgrade)

#### ty

`ty` is Astral's extremely fast Python type checker (written in Rust), replacing mypy. It is 10–60× faster than mypy and includes a full LSP for editor integration.

```bash
# Type-check the entire package
uv run ty check sar/

# Check a specific file
uv run ty check sar/models/vit.py

# Explain a specific diagnostic rule
uv run ty explain rule <rule-name>

# Run without installing (one-off)
uvx ty check sar/
```

Configuration lives under `[tool.ty]` in `pyproject.toml`. Python version is inferred automatically from `requires-python = ">=3.13"`.

To suppress a specific diagnostic inline:
```python
x: int = some_func()  # type: ignore[assignment]
```

Rule levels can be set per-rule:
```toml
[tool.ty.rules]
possibly-missing-import = "warn"   # or "error" | "ignore"
```

### Patterns to Follow

- Use `pathlib.Path` for file paths (not `os.path`)
- Device handling: `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Checkpoints: save encoder separately from full model for easy reuse
- Include an `if __name__ == "__main__":` block in each module for smoke testing
- Training scripts accept arguments as function parameters — avoid hardcoded paths

### Ignored by Git

```
__pycache__/
.venv/
Airport_Dataset_v0_*/   ← raw dataset
dataset/                ← split dataset output
*.zip
```

---

## Testing

There is no formal test suite. Each module includes inline smoke tests in `if __name__ == "__main__"` blocks. Run them with:

```bash
uv run python -m sar.models.vit       # Tests ViT forward pass
uv run python -m sar.models.mae       # Tests MAE with random images
uv run python -m sar.models.simclr    # Tests SimCLR with dual views
uv run python -m sar.models.rtdetr    # Tests RT-DETR with dummy targets
uv run python -m sar.augmentation     # Tests all augmentation pipelines
uv run python -m sar.data.datasets    # Tests dataset classes and DataLoader
```

---

## Common Tasks for AI Assistants

### Adding a New Model

1. Create `sar/models/<model_name>.py`
2. Implement with `if __name__ == "__main__"` smoke test
3. Export from `sar/models/__init__.py` if needed
4. Write a matching training script in `sar/train/train_<model_name>.py`

### Modifying Augmentations

Edit `sar/augmentation.py`. Each stage has a dedicated function. Use `albumentations` for all transforms — bbox-safe pipelines must use `BboxParams(format='yolo', label_fields=['class_labels'])`.

### Changing Dataset Splits

Edit `main.py` arguments (ratios, seed, paths). The split logic in `sar/dataset.py` groups images by airport prefix to avoid leakage — preserve this behavior.

### Debugging Training

- Check TensorBoard: `tensorboard --logdir runs/`
- Reduce batch size if CUDA OOM (detection: batch=8 needs ~4GB VRAM)
- Encoder freezing helps early detection training stability

---

## Hardware Requirements

| Task | VRAM |
|---|---|
| MAE pretraining | ~2GB |
| SimCLR pretraining | ~2GB |
| Detection training (batch=8) | ~4GB |

CPU training is supported but very slow. GPU (CUDA) is strongly recommended.
