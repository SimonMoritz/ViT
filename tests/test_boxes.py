"""Tests for bounding box utility functions."""

import torch
import pytest

from sar.utils.boxes import (
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    box_iou,
    generalized_box_iou,
    giou_loss,
)


def test_cxcywh_to_xyxy_roundtrip():
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.3, 0.7, 0.2, 0.1]])
    xyxy = box_cxcywh_to_xyxy(boxes)
    back = box_xyxy_to_cxcywh(xyxy)
    assert torch.allclose(boxes, back, atol=1e-6)


def test_cxcywh_to_xyxy_values():
    box = torch.tensor([[0.5, 0.5, 0.4, 0.6]])
    xyxy = box_cxcywh_to_xyxy(box)
    expected = torch.tensor([[0.3, 0.2, 0.7, 0.8]])
    assert torch.allclose(xyxy, expected, atol=1e-6)


def test_box_iou_identical():
    boxes = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
    iou = box_iou(boxes, boxes)
    assert torch.allclose(iou, torch.ones(1, 1), atol=1e-5)


def test_box_iou_no_overlap():
    b1 = torch.tensor([[0.0, 0.0, 0.4, 0.4]])
    b2 = torch.tensor([[0.6, 0.6, 1.0, 1.0]])
    iou = box_iou(b1, b2)
    assert iou.item() == pytest.approx(0.0, abs=1e-6)


def test_giou_loss_perfect():
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    loss = giou_loss(boxes, boxes)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)
