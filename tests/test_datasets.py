from pathlib import Path

import torch
from PIL import Image

from sar.data.datasets import DetectionDataset, PretrainDataset, collate_fn_detection


def _write_rgb_image(path: Path) -> None:
    Image.new("RGB", (32, 32), color=(128, 128, 128)).save(path)


def test_pretrain_dataset_loads_images(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_rgb_image(image_dir / "sample.jpg")

    dataset = PretrainDataset(image_dir)
    image = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 32, 32)


def test_detection_dataset_and_collate(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    for index in range(2):
        _write_rgb_image(image_dir / f"sample_{index}.jpg")
        (label_dir / f"sample_{index}.txt").write_text("0 0.5 0.5 0.25 0.25\n")

    dataset = DetectionDataset(image_dir, label_dir, return_dict=True)
    first_item = dataset[0]
    images, targets = collate_fn_detection([first_item, dataset[1]])

    assert images.shape == (2, 3, 32, 32)
    assert len(targets) == 2
    assert targets[0]["boxes"].shape == (1, 4)

