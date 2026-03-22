"""Bounding box conversion and overlap utilities."""

import torch
from torch import Tensor


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack((x1, y1, x2, y2), dim=-1)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h)."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = x2 - x1
    h = y2 - y1
    return torch.stack((cx, cy, w, h), dim=-1)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute pairwise IoU for boxes in (x1, y1, x2, y2) format."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(
        min=0
    )
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(
        min=0
    )

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute pairwise generalized IoU for boxes in (cx, cy, w, h) format."""
    boxes1_xyxy = box_cxcywh_to_xyxy(boxes1)
    boxes2_xyxy = box_cxcywh_to_xyxy(boxes2)

    iou = box_iou(boxes1_xyxy, boxes2_xyxy)

    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
    rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]).clamp(min=0) * (
        boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1]
    ).clamp(min=0)
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]).clamp(min=0) * (
        boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1]
    ).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter

    enclosure_lt = torch.min(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
    enclosure_rb = torch.max(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
    enclosure_wh = (enclosure_rb - enclosure_lt).clamp(min=0)
    enclosure_area = enclosure_wh[:, :, 0] * enclosure_wh[:, :, 1]

    return iou - (enclosure_area - union) / enclosure_area.clamp(min=1e-6)


def giou_loss(pred_boxes: Tensor, target_boxes: Tensor) -> Tensor:
    """Compute mean GIoU loss for matched boxes in (cx, cy, w, h) format."""
    giou = torch.diagonal(generalized_box_iou(pred_boxes, target_boxes))
    return (1 - giou).mean()
