"""Bounding box utility functions shared across the project.

All boxes are in normalised [0, 1] coordinate space unless noted otherwise.
Formats:
    cxcywh  - (centre_x, centre_y, width, height)
    xyxy    - (x1, y1, x2, y2)  top-left and bottom-right corners
"""

import torch

from sar.config import IOU_EPS


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        boxes: (..., 4) tensor in cxcywh format
    Returns:
        (..., 4) tensor in xyxy format
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        boxes: (..., 4) tensor in xyxy format
    Returns:
        (..., 4) tensor in cxcywh format
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes (xyxy format).

    Args:
        boxes1: (N, 4)
        boxes2: (M, 4)
    Returns:
        (N, M) IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=IOU_EPS)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise GIoU between two sets of boxes (cxcywh format).

    Args:
        boxes1: (N, 4) in cxcywh format
        boxes2: (M, 4) in cxcywh format
    Returns:
        (N, M) GIoU matrix
    """
    boxes1_xyxy = box_cxcywh_to_xyxy(boxes1)
    boxes2_xyxy = box_cxcywh_to_xyxy(boxes2)

    iou = box_iou(boxes1_xyxy, boxes2_xyxy)

    # Enclosing box
    lt = torch.min(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
    rb = torch.max(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    area_c = wh[:, :, 0] * wh[:, :, 1]

    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
    union = area1[:, None] + area2[None, :] - iou * area1[:, None]

    return iou - (area_c - union) / area_c.clamp(min=IOU_EPS)


def giou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """GIoU loss for matched pairs of boxes (cxcywh format).

    Args:
        pred_boxes: (N, 4) predictions in cxcywh format
        target_boxes: (N, 4) targets in cxcywh format
    Returns:
        Scalar mean GIoU loss
    """
    giou = torch.diagonal(generalized_box_iou(pred_boxes, target_boxes))
    return (1 - giou).mean()
