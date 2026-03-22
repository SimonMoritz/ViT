import torch

from sar.models.mae import MAE
from sar.models.rtdetr import RTDETR, RTDETRLoss
from sar.models.simclr import SimCLR
from sar.models.vit import ViTTiny


def test_vit_tiny_output_shape() -> None:
    model = ViTTiny(img_size=224, use_cls_token=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, (224 // 16) ** 2, model.embed_dim)


def test_mae_forward_shapes() -> None:
    encoder = ViTTiny(img_size=224, use_cls_token=False)
    model = MAE(encoder, mask_ratio=0.75, patch_size=16)
    x = torch.randn(2, 3, 224, 224)
    loss, pred, mask = model(x)
    assert loss.ndim == 0
    assert pred.shape == x.shape
    assert mask.shape == (2, (224 // 16) ** 2)


def test_simclr_forward_shapes() -> None:
    encoder = ViTTiny(img_size=224, use_cls_token=False)
    model = SimCLR(encoder, projection_dim=128, temperature=0.5)
    x1 = torch.randn(4, 3, 224, 224)
    x2 = torch.randn(4, 3, 224, 224)
    loss, z1, z2 = model(x1, x2)
    assert loss.ndim == 0
    assert z1.shape == (4, 128)
    assert z2.shape == (4, 128)


def test_rtdetr_forward_and_loss() -> None:
    encoder = ViTTiny(img_size=224, use_cls_token=False)
    model = RTDETR(encoder, num_classes=1, num_queries=100)
    x = torch.randn(2, 3, 224, 224)
    pred_logits, pred_boxes = model(x)

    targets = [
        {
            "labels": torch.tensor([0, 0]),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.1, 0.2]]),
        },
        {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.6, 0.6, 0.15, 0.25]])},
    ]

    losses = RTDETRLoss(num_classes=1)(pred_logits, pred_boxes, targets)

    assert pred_logits.shape == (2, 100, 2)
    assert pred_boxes.shape == (2, 100, 4)
    assert {"loss", "loss_ce", "loss_bbox", "loss_giou"} <= losses.keys()

