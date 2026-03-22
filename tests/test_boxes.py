import torch

from sar.utils.boxes import (
    box_cxcywh_to_xyxy,
    box_iou,
    box_xyxy_to_cxcywh,
    generalized_box_iou,
    giou_loss,
)


def test_box_format_round_trip() -> None:
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.2, 0.3, 0.1, 0.1]])
    xyxy_boxes = box_cxcywh_to_xyxy(boxes)
    restored = box_xyxy_to_cxcywh(xyxy_boxes)
    assert torch.allclose(restored, boxes)


def test_box_iou_identical_boxes_equal_one() -> None:
    boxes = torch.tensor([[0.1, 0.2, 0.5, 0.7]])
    iou = box_iou(boxes, boxes)
    assert torch.allclose(iou, torch.ones_like(iou))


def test_generalized_box_iou_penalizes_non_overlap() -> None:
    boxes1 = torch.tensor([[0.25, 0.25, 0.2, 0.2]])
    boxes2 = torch.tensor([[0.75, 0.75, 0.2, 0.2]])
    giou = generalized_box_iou(boxes1, boxes2)
    assert giou.item() < 0


def test_giou_loss_zero_for_perfect_match() -> None:
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]])
    loss = giou_loss(boxes, boxes)
    assert torch.isclose(loss, torch.tensor(0.0))

