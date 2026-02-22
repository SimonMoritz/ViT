"""Shared training utilities used across MAE, SimCLR, and detection training."""

import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device


def setup_output_dir(output_dir: str | Path) -> Path:
    """Create output directory (including parents) and return as Path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_tensorboard_writer(output_dir: Path) -> SummaryWriter:
    """Create a TensorBoard SummaryWriter logging to output_dir/logs."""
    return SummaryWriter(output_dir / "logs")


def build_cosine_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    num_epochs: int,
    betas: tuple[float, float] = (0.9, 0.999),
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Create an AdamW optimizer with a cosine annealing schedule.

    Args:
        model: Model whose parameters will be optimised.
        lr: Peak learning rate.
        weight_decay: Weight decay coefficient.
        num_epochs: Total training epochs (used as T_max for cosine schedule).
        betas: Adam beta coefficients.

    Returns:
        (optimizer, scheduler) tuple.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=lr * 0.01,
    )
    return optimizer, scheduler


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    **extra_scalars,
) -> None:
    """Save a training checkpoint.

    Args:
        path: File path to save to.
        epoch: Current epoch number (1-based).
        model: Model to checkpoint.
        optimizer: Optimizer state to checkpoint.
        scheduler: LR scheduler state to checkpoint.
        **extra_scalars: Additional scalar values to store (e.g. loss=0.42).
    """
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            **extra_scalars,
        },
        path,
    )
    print(f"Saved checkpoint: {path}")


def log_step_scalars(
    writer: SummaryWriter,
    scalars: dict,
    global_step: int,
    log_every: int = 10,
) -> None:
    """Write scalar summaries to TensorBoard every `log_every` steps.

    Args:
        writer: Active SummaryWriter.
        scalars: Mapping of tag → scalar value.
        global_step: Current global training step.
        log_every: Logging frequency in steps.
    """
    if global_step % log_every == 0:
        for tag, value in scalars.items():
            writer.add_scalar(tag, value, global_step)
