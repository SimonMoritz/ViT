"""Training script for MAE pretraining."""

import argparse
from pathlib import Path

from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from sar.augmentation import get_pretrain_augmentation
from sar.config import (
    DEFAULT_IMG_SIZE,
    DEFAULT_NUM_WORKERS,
    LOG_IMAGE_EVERY,
    LOG_SCALAR_EVERY,
    MAE_ADAMW_BETAS,
    MAE_BATCH_SIZE,
    MAE_DECODER_UPSCALE_IMG_SIZE,
    MAE_LARGE_DECODER_DEPTH,
    MAE_LARGE_DECODER_EMBED_DIM,
    MAE_LARGE_DECODER_HEADS,
    MAE_LR,
    MAE_MASK_RATIO,
    MAE_NUM_EPOCHS,
    MAE_OUTPUT_DIR,
    MAE_SAVE_EVERY,
    MAE_STANDARD_DECODER_DEPTH,
    MAE_STANDARD_DECODER_EMBED_DIM,
    MAE_STANDARD_DECODER_HEADS,
    MAE_WEIGHT_DECAY,
    RAW_IMAGE_DIR,
    VIT_PATCH_SIZE,
)
from sar.data.datasets import PretrainDataset
from sar.models.mae import MAE
from sar.models.vit import ViTTiny
from sar.train.trainer import (
    build_cosine_optimizer,
    build_tensorboard_writer,
    get_device,
    log_step_scalars,
    save_checkpoint,
    setup_output_dir,
)

DecoderConfig = dict[str, int]


def train_mae(
    img_dir: str | Path = RAW_IMAGE_DIR,
    output_dir: str | Path = MAE_OUTPUT_DIR,
    img_size: int = DEFAULT_IMG_SIZE,
    batch_size: int = MAE_BATCH_SIZE,
    num_epochs: int = MAE_NUM_EPOCHS,
    lr: float = MAE_LR,
    weight_decay: float = MAE_WEIGHT_DECAY,
    mask_ratio: float = MAE_MASK_RATIO,
    num_workers: int = DEFAULT_NUM_WORKERS,
    save_every: int = MAE_SAVE_EVERY,
) -> None:
    """
    Train MAE for self-supervised pretraining.

    Args:
        img_dir: Directory with all images (train + val + test for pretraining)
        output_dir: Directory to save checkpoints
        img_size: Image size
        batch_size: Batch size
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        mask_ratio: Ratio of patches to mask
        num_workers: Number of dataloader workers
        save_every: Save checkpoint every N epochs
    """
    device = get_device()
    print(f"Using device: {device}")

    output_dir = setup_output_dir(output_dir)
    writer = build_tensorboard_writer(output_dir)

    # Dataset and dataloader
    transform = get_pretrain_augmentation(img_size)
    dataset = PretrainDataset(img_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    # Model
    encoder = ViTTiny(img_size=img_size, use_cls_token=False)

    # Scale decoder capacity with image size
    if img_size >= MAE_DECODER_UPSCALE_IMG_SIZE:
        decoder_config: DecoderConfig = {
            "decoder_embed_dim": MAE_LARGE_DECODER_EMBED_DIM,
            "decoder_depth": MAE_LARGE_DECODER_DEPTH,
            "decoder_n_heads": MAE_LARGE_DECODER_HEADS,
        }
        print(
            f"Using LARGE decoder for {img_size}px: dim={MAE_LARGE_DECODER_EMBED_DIM}, "
            f"depth={MAE_LARGE_DECODER_DEPTH}, heads={MAE_LARGE_DECODER_HEADS}"
        )
    else:
        decoder_config = {
            "decoder_embed_dim": MAE_STANDARD_DECODER_EMBED_DIM,
            "decoder_depth": MAE_STANDARD_DECODER_DEPTH,
            "decoder_n_heads": MAE_STANDARD_DECODER_HEADS,
        }
        print(
            f"Using standard decoder for {img_size}px: dim={MAE_STANDARD_DECODER_EMBED_DIM}, "
            f"depth={MAE_STANDARD_DECODER_DEPTH}, heads={MAE_STANDARD_DECODER_HEADS}"
        )

    mae = MAE(encoder, mask_ratio=mask_ratio, patch_size=VIT_PATCH_SIZE, **decoder_config)
    mae = mae.to(device)

    print(f"Model parameters: {sum(p.numel() for p in mae.parameters()) / 1e6:.2f}M")

    # Optimizer (AdamW with cosine schedule as in MAE paper)
    optimizer, scheduler = build_cosine_optimizer(
        mae.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        betas=MAE_ADAMW_BETAS,
    )

    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        mae.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for _batch_idx, images in enumerate(pbar):
            images = images.to(device)
            if not isinstance(images, Tensor):
                raise TypeError("MAE pretraining expects tensor batches from PretrainDataset.")

            # Forward pass
            loss, pred_imgs, mask = mae(images)

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

            # Log reconstructions
            if global_step % LOG_IMAGE_EVERY == 0:
                original = images[0].cpu()
                reconstructed = pred_imgs[0].cpu()
                writer.add_image("mae/original", original, global_step)
                writer.add_image("mae/reconstructed", reconstructed.clamp(0, 1), global_step)

        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        # Learning rate schedule step
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            checkpoint_path = output_dir / f"mae_epoch_{epoch + 1}.pth"
            save_checkpoint(
                checkpoint_path,
                {
                    "epoch": epoch + 1,
                    "model_state_dict": mae.state_dict(),
                    "encoder_state_dict": mae.encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                },
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final encoder
    encoder_path = output_dir / "encoder_mae_final.pth"
    save_checkpoint(encoder_path, mae.encoder.state_dict())
    print(f"Saved final encoder: {encoder_path}")

    writer.close()
    print("MAE training complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAE for self-supervised pretraining")
    parser.add_argument(
        "--img_dir", type=str, default=str(RAW_IMAGE_DIR), help="Directory with images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(MAE_OUTPUT_DIR),
        help="Output directory for checkpoints",
    )
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE, help="Image size")
    parser.add_argument("--batch_size", type=int, default=MAE_BATCH_SIZE, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=MAE_NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=MAE_LR, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=MAE_WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--mask_ratio", type=float, default=MAE_MASK_RATIO, help="Mask ratio")
    parser.add_argument(
        "--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=MAE_SAVE_EVERY,
        help="Save checkpoint every N epochs",
    )

    args = parser.parse_args()

    train_mae(
        img_dir=args.img_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        mask_ratio=args.mask_ratio,
        num_workers=args.num_workers,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
