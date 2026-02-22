"""Smoke tests for model forward passes and parameter counts."""

import torch
import pytest

from sar.models.vit import ViTTiny
from sar.models.mae import MAE
from sar.models.simclr import SimCLR
from sar.models.rtdetr import RTDETR, RTDETRLoss


def test_vit_tiny_no_cls():
    model = ViTTiny(img_size=224, use_cls_token=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 196, 192), f"Unexpected shape: {out.shape}"


def test_vit_tiny_with_cls():
    model = ViTTiny(img_size=224, use_cls_token=True)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 197, 192), f"Unexpected shape: {out.shape}"
    cls = model.get_cls_token(x)
    assert cls.shape == (2, 192)


def test_vit_tiny_param_count():
    model = ViTTiny(img_size=224, use_cls_token=False)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    assert n_params < 6.0, f"Too many params: {n_params:.2f}M"


def test_mae_forward():
    encoder = ViTTiny(img_size=224, use_cls_token=False)
    mae = MAE(encoder, mask_ratio=0.75, patch_size=16)
    x = torch.randn(2, 3, 224, 224)
    loss, pred, mask = mae(x)
    assert loss.ndim == 0, "Loss should be scalar"
    assert pred.shape == x.shape, f"Pred shape {pred.shape} != input shape {x.shape}"
    assert mask.shape == (2, 196), f"Unexpected mask shape: {mask.shape}"
    assert 0.7 < mask.float().mean().item() < 0.8, "Mask ratio should be ~0.75"


def test_simclr_forward():
    encoder = ViTTiny(img_size=224, use_cls_token=False)
    simclr = SimCLR(encoder, projection_dim=128, temperature=0.5)
    x1 = torch.randn(4, 3, 224, 224)
    x2 = torch.randn(4, 3, 224, 224)
    loss, z1, z2 = simclr(x1, x2)
    assert loss.ndim == 0
    assert z1.shape == (4, 128)
    assert z2.shape == (4, 128)


def test_rtdetr_forward():
    encoder = ViTTiny(img_size=224, use_cls_token=False)
    rtdetr = RTDETR(encoder, num_classes=1, num_queries=100)
    x = torch.randn(2, 3, 224, 224)
    pred_logits, pred_boxes = rtdetr(x)
    assert pred_logits.shape == (2, 100, 2)
    assert pred_boxes.shape == (2, 100, 4)
    # Boxes should be in [0, 1] after sigmoid
    assert pred_boxes.min() >= 0.0 and pred_boxes.max() <= 1.0


def test_rtdetr_loss():
    encoder = ViTTiny(img_size=224, use_cls_token=False)
    rtdetr = RTDETR(encoder, num_classes=1, num_queries=100)
    x = torch.randn(2, 3, 224, 224)
    pred_logits, pred_boxes = rtdetr(x)

    targets = [
        {
            'labels': torch.tensor([0, 0]),
            'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.1, 0.2]]),
        },
        {
            'labels': torch.tensor([0]),
            'boxes': torch.tensor([[0.6, 0.6, 0.15, 0.25]]),
        },
    ]

    criterion = RTDETRLoss(num_classes=1)
    losses = criterion(pred_logits, pred_boxes, targets)
    assert 'loss' in losses
    assert losses['loss'].ndim == 0
    assert losses['loss'].item() >= 0.0
