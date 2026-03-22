"""Shared training helpers."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from sar.config import COSINE_ETA_MIN_RATIO


def get_device() -> torch.device:
    """Return the preferred training device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_output_dir(output_dir: str | Path) -> Path:
    """Create and return the checkpoint output directory."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_tensorboard_writer(output_dir: str | Path) -> SummaryWriter:
    """Create a TensorBoard writer rooted at `<output_dir>/logs`."""
    return SummaryWriter(Path(output_dir) / "logs")


def build_cosine_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
    num_epochs: int,
    betas: tuple[float, float] = (0.9, 0.999),
) -> tuple[Optimizer, CosineAnnealingLR]:
    """Build an AdamW optimizer with cosine LR decay."""
    optimizer = AdamW(parameters, lr=lr, betas=betas, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * COSINE_ETA_MIN_RATIO)
    return optimizer, scheduler


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Persist a checkpoint payload."""
    torch.save(dict(payload), Path(path))


def log_step_scalars(writer: SummaryWriter, scalars: Mapping[str, float], step: int) -> None:
    """Log a batch of scalar values at the same global step."""
    for name, value in scalars.items():
        writer.add_scalar(name, value, step)

