"""Centralized constants for the SAR airport detection project.

Import from here instead of hardcoding values throughout the codebase.
"""

# ---------------------------------------------------------------------------
# Image normalization
# SAR images are normalised to [-1, 1] using these statistics.
# ---------------------------------------------------------------------------
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

# ---------------------------------------------------------------------------
# ViT-Tiny architecture defaults
# ---------------------------------------------------------------------------
IMG_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 192
VIT_DEPTH = 12
VIT_HEADS = 3
VIT_MLP_RATIO = 4.0

# ---------------------------------------------------------------------------
# MAE defaults
# ---------------------------------------------------------------------------
MAE_MASK_RATIO = 0.75
MAE_DECODER_EMBED_DIM = 128
MAE_DECODER_DEPTH = 4
MAE_DECODER_HEADS = 4

# Large-decoder config used when img_size >= 512
MAE_LARGE_DECODER_EMBED_DIM = 256
MAE_LARGE_DECODER_DEPTH = 8
MAE_LARGE_DECODER_HEADS = 8

# ---------------------------------------------------------------------------
# SimCLR defaults
# ---------------------------------------------------------------------------
SIMCLR_TEMPERATURE = 0.5
SIMCLR_PROJECTION_DIM = 128
SIMCLR_PROJECTION_HIDDEN_DIM = 512

# ---------------------------------------------------------------------------
# RT-DETR defaults
# ---------------------------------------------------------------------------
RTDETR_NUM_QUERIES = 100
RTDETR_HIDDEN_DIM = 256
RTDETR_NUM_DECODER_LAYERS = 6
RTDETR_DIM_FEEDFORWARD = 1024

# Loss weights
RTDETR_LOSS_WEIGHT_BBOX = 5.0
RTDETR_LOSS_WEIGHT_GIOU = 2.0

# ---------------------------------------------------------------------------
# Dataset defaults
# ---------------------------------------------------------------------------
DEFAULT_IMG_DIR = "Airport_Dataset_v0_images"
DEFAULT_LABEL_DIR = "Airport_Dataset_v0_labels"
DEFAULT_DATASET_OUTPUT_DIR = "dataset"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------
TRUNC_NORMAL_STD = 0.02

# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------
IOU_EPS = 1e-6
