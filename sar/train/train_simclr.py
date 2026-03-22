"""Training script for SimCLR pretraining."""

import argparse
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from sar.augmentation import DualViewTransform, get_simclr_augmentation
from sar.config import (
    DEFAULT_IMG_SIZE,
    DEFAULT_NUM_WORKERS,
    LOG_IMAGE_EVERY,
    LOG_SCALAR_EVERY,
    RAW_IMAGE_DIR,
    SIMCLR_ADAMW_BETAS,
    SIMCLR_BATCH_SIZE,
    SIMCLR_LR,
    SIMCLR_NUM_EPOCHS,
    SIMCLR_OUTPUT_DIR,
    SIMCLR_PROJECTION_DIM,
    SIMCLR_SAVE_EVERY,
    SIMCLR_TEMPERATURE,
    SIMCLR_WEIGHT_DECAY,
)
from sar.data.datasets import PretrainDataset
from sar.models.simclr import SimCLR
from sar.models.vit import ViTTiny
from sar.train.trainer import (
    build_cosine_optimizer,
    build_tensorboard_writer,
    get_device,
    log_step_scalars,
    save_checkpoint,
    setup_output_dir,
)


def train_simclr(
    img_dir: str | Path = RAW_IMAGE_DIR,
    output_dir: str | Path = SIMCLR_OUTPUT_DIR,
    encoder_pretrain_path: str | Path | None = None,  # Optional: load MAE pretrained encoder
    img_size: int = DEFAULT_IMG_SIZE,
    batch_size: int = SIMCLR_BATCH_SIZE,
    num_epochs: int = SIMCLR_NUM_EPOCHS,
    lr: float = SIMCLR_LR,
    weight_decay: float = SIMCLR_WEIGHT_DECAY,
    temperature: float = SIMCLR_TEMPERATURE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    save_every: int = SIMCLR_SAVE_EVERY,
) -> None:
    """
    Train SimCLR for contrastive self-supervised learning.

    Args:
        img_dir: Directory with all images
        output_dir: Directory to save checkpoints
        encoder_pretrain_path: Optional path to MAE pretrained encoder
        img_size: Image size
        batch_size: Batch size
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        temperature: Temperature for contrastive loss
        num_workers: Number of dataloader workers
        save_every: Save checkpoint every N epochs
    """
    device = get_device()
    print(f"Using device: {device}")

    output_dir = setup_output_dir(output_dir)
    writer = build_tensorboard_writer(output_dir)

    # Dataset and dataloader (dual view transform)
    transform = DualViewTransform(get_simclr_augmentation(img_size))
    dataset = PretrainDataset(img_dir, transform=transform)

    # Custom collate function for dual views
    def collate_fn(batch: list[tuple[Tensor, Tensor] | Tensor]) -> tuple[Tensor, Tensor]:
        dual_views = [cast(tuple[Tensor, Tensor], item) for item in batch]
        view1 = torch.stack([item[0] for item in dual_views])
        view2 = torch.stack([item[1] for item in dual_views])
        return view1, view2

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    # Model
    encoder = ViTTiny(img_size=img_size, use_cls_token=False)

    # Load MAE pretrained weights if provided
    if encoder_pretrain_path is not None:
        print(f"Loading pretrained encoder from {encoder_pretrain_path}")
        state_dict = torch.load(encoder_pretrain_path, map_location="cpu")
        encoder.load_state_dict(state_dict)

    simclr = SimCLR(encoder, projection_dim=SIMCLR_PROJECTION_DIM, temperature=temperature)
    simclr = simclr.to(device)

    print(f"Model parameters: {sum(p.numel() for p in simclr.parameters()) / 1e6:.2f}M")

    # Optimizer
    optimizer, scheduler = build_cosine_optimizer(
        simclr.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        betas=SIMCLR_ADAMW_BETAS,
    )

    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        simclr.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for _batch_idx, (view1, view2) in enumerate(pbar):
            view1 = view1.to(device)
            view2 = view2.to(device)

            # Forward pass
            loss, _z1, _z2 = simclr(view1, view2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            epoch_loss += loss.item()
            global_step += 1

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"}
            )

            # TensorBoard logging
            if global_step % LOG_SCALAR_EVERY == 0:
                log_step_scalars(
                    writer,
                    {
                        "train/loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    global_step,
                )

            # Log sample views
            if global_step % LOG_IMAGE_EVERY == 0:
                writer.add_image("simclr/view1", view1[0].cpu(), global_step)
                writer.add_image("simclr/view2", view2[0].cpu(), global_step)

        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        # Learning rate schedule step
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            checkpoint_path = output_dir / f"simclr_epoch_{epoch + 1}.pth"
            save_checkpoint(
                checkpoint_path,
                {
                    "epoch": epoch + 1,
                    "model_state_dict": simclr.state_dict(),
                    "encoder_state_dict": simclr.encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                },
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final encoder
    encoder_path = output_dir / "encoder_simclr_final.pth"
    save_checkpoint(encoder_path, simclr.encoder.state_dict())
    print(f"Saved final encoder: {encoder_path}")

    writer.close()
    print("SimCLR training complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SimCLR for contrastive learning")
    parser.add_argument(
        "--img_dir", type=str, default=str(RAW_IMAGE_DIR), help="Directory with images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(SIMCLR_OUTPUT_DIR),
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--encoder_pretrain_path",
        type=str,
        default=None,
        help="Path to MAE pretrained encoder (optional)",
    )
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE, help="Image size")
    parser.add_argument("--batch_size", type=int, default=SIMCLR_BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--num_epochs", type=int, default=SIMCLR_NUM_EPOCHS, help="Number of epochs"
    )
    parser.add_argument("--lr", type=float, default=SIMCLR_LR, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=SIMCLR_WEIGHT_DECAY, help="Weight decay"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=SIMCLR_TEMPERATURE,
        help="Temperature for contrastive loss",
    )
    parser.add_argument(
        "--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=SIMCLR_SAVE_EVERY,
        help="Save checkpoint every N epochs",
    )

    args = parser.parse_args()

    train_simclr(
        img_dir=args.img_dir,
        output_dir=args.output_dir,
        encoder_pretrain_path=args.encoder_pretrain_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        num_workers=args.num_workers,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
