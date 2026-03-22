"""Training script for RT-DETR object detection."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sar.augmentation import get_detection_train_augmentation, get_detection_val_augmentation
from sar.config import (
    DEFAULT_IMG_SIZE,
    DEFAULT_NUM_WORKERS,
    DETECTION_ADAMW_BETAS,
    DETECTION_BATCH_SIZE,
    DETECTION_FREEZE_ENCODER_EPOCHS,
    DETECTION_GRAD_CLIP_NORM,
    DETECTION_LR,
    DETECTION_NUM_CLASSES,
    DETECTION_NUM_EPOCHS,
    DETECTION_NUM_QUERIES,
    DETECTION_OUTPUT_DIR,
    DETECTION_SAVE_EVERY,
    DETECTION_WEIGHT_DECAY,
    LOG_SCALAR_EVERY,
    TRAIN_IMAGE_DIR,
    TRAIN_LABEL_DIR,
    VAL_IMAGE_DIR,
    VAL_LABEL_DIR,
)
from sar.data.datasets import DetectionDataset, TensorDict, collate_fn_detection
from sar.models.rtdetr import RTDETR, RTDETRLoss
from sar.models.vit import ViTTiny
from sar.train.trainer import (
    build_cosine_optimizer,
    build_tensorboard_writer,
    get_device,
    log_step_scalars,
    save_checkpoint,
    setup_output_dir,
)


def train_detection(
    train_img_dir: str | Path = TRAIN_IMAGE_DIR,
    train_label_dir: str | Path = TRAIN_LABEL_DIR,
    val_img_dir: str | Path = VAL_IMAGE_DIR,
    val_label_dir: str | Path = VAL_LABEL_DIR,
    output_dir: str | Path = DETECTION_OUTPUT_DIR,
    encoder_pretrain_path: str | Path | None = None,  # Path to pretrained encoder (MAE or SimCLR)
    img_size: int = DEFAULT_IMG_SIZE,
    batch_size: int = DETECTION_BATCH_SIZE,
    num_epochs: int = DETECTION_NUM_EPOCHS,
    lr: float = DETECTION_LR,
    weight_decay: float = DETECTION_WEIGHT_DECAY,
    num_queries: int = DETECTION_NUM_QUERIES,
    num_workers: int = DEFAULT_NUM_WORKERS,
    freeze_encoder_epochs: int = DETECTION_FREEZE_ENCODER_EPOCHS,
    save_every: int = DETECTION_SAVE_EVERY,
) -> None:
    """
    Train RT-DETR for object detection.

    Args:
        train_img_dir: Training images directory
        train_label_dir: Training labels directory
        val_img_dir: Validation images directory
        val_label_dir: Validation labels directory
        output_dir: Output directory for checkpoints
        encoder_pretrain_path: Path to pretrained encoder
        img_size: Image size
        batch_size: Batch size
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        num_queries: Number of detection queries
        num_workers: Number of dataloader workers
        freeze_encoder_epochs: Freeze encoder for first N epochs
        save_every: Save checkpoint every N epochs
    """
    device = get_device()
    print(f"Using device: {device}")

    output_dir = setup_output_dir(output_dir)
    writer = build_tensorboard_writer(output_dir)

    # Datasets
    train_transform = get_detection_train_augmentation(img_size)
    val_transform = get_detection_val_augmentation(img_size)

    train_dataset = DetectionDataset(
        train_img_dir, train_label_dir, transform=train_transform, return_dict=True
    )
    val_dataset = DetectionDataset(
        val_img_dir, val_label_dir, transform=val_transform, return_dict=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_detection,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_detection,
    )

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")
    print(f"Train batches per epoch: {len(train_loader)}")

    # Model
    encoder = ViTTiny(img_size=img_size, use_cls_token=False)

    # Load pretrained encoder if provided
    if encoder_pretrain_path is not None:
        print(f"Loading pretrained encoder from {encoder_pretrain_path}")
        state_dict = torch.load(encoder_pretrain_path, map_location="cpu")
        encoder.load_state_dict(state_dict)
        print("Pretrained encoder loaded successfully!")

    rtdetr = RTDETR(encoder, num_classes=DETECTION_NUM_CLASSES, num_queries=num_queries)
    rtdetr = rtdetr.to(device)

    print(f"Model parameters: {sum(p.numel() for p in rtdetr.parameters()) / 1e6:.2f}M")

    # Loss
    criterion = RTDETRLoss(num_classes=DETECTION_NUM_CLASSES)

    optimizer, scheduler = build_cosine_optimizer(
        rtdetr.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        betas=DETECTION_ADAMW_BETAS,
    )

    # Training loop
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Freeze/unfreeze encoder
        if freeze_encoder_epochs > 0 and epoch < freeze_encoder_epochs:
            print(f"Encoder frozen (epoch {epoch + 1}/{freeze_encoder_epochs})")
            for param in rtdetr.encoder.parameters():
                param.requires_grad = False
        else:
            for param in rtdetr.encoder.parameters():
                param.requires_grad = True

        # Training
        rtdetr.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")

        for _batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            typed_targets: list[TensorDict] = [
                {k: v.to(device) for k, v in t.items()} for t in targets
            ]

            # Forward pass
            pred_logits, pred_boxes = rtdetr(images)
            losses = criterion(pred_logits, pred_boxes, typed_targets)

            loss = losses["loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rtdetr.parameters(), max_norm=DETECTION_GRAD_CLIP_NORM)
            optimizer.step()

            # Logging
            train_loss += loss.item()
            global_step += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "loss_ce": f"{losses['loss_ce'].item():.4f}",
                    "loss_bbox": f"{losses['loss_bbox'].item():.4f}",
                    "loss_giou": f"{losses['loss_giou'].item():.4f}",
                }
            )

            # TensorBoard logging
            if global_step % LOG_SCALAR_EVERY == 0:
                log_step_scalars(
                    writer,
                    {
                        "train/loss": loss.item(),
                        "train/loss_ce": losses["loss_ce"].item(),
                        "train/loss_bbox": losses["loss_bbox"].item(),
                        "train/loss_giou": losses["loss_giou"].item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    global_step,
                )

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}")

        # Validation
        rtdetr.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
                images = images.to(device)
                typed_targets: list[TensorDict] = [
                    {k: v.to(device) for k, v in t.items()} for t in targets
                ]

                pred_logits, pred_boxes = rtdetr(images)
                losses = criterion(pred_logits, pred_boxes, typed_targets)

                val_loss += losses["loss"].item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} - Val Loss: {avg_val_loss:.4f}")

        writer.add_scalar("val/loss", avg_val_loss, epoch)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = output_dir / "best_model.pth"
            save_checkpoint(
                best_model_path,
                {
                    "epoch": epoch + 1,
                    "model_state_dict": rtdetr.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                },
            )
            print(f"Saved best model: {best_model_path}")

        # Learning rate schedule
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            save_checkpoint(
                checkpoint_path,
                {
                    "epoch": epoch + 1,
                    "model_state_dict": rtdetr.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    writer.close()
    print("Detection training complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RT-DETR for object detection")
    parser.add_argument("--train_img_dir", type=str, default=str(TRAIN_IMAGE_DIR))
    parser.add_argument("--train_label_dir", type=str, default=str(TRAIN_LABEL_DIR))
    parser.add_argument("--val_img_dir", type=str, default=str(VAL_IMAGE_DIR))
    parser.add_argument("--val_label_dir", type=str, default=str(VAL_LABEL_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DETECTION_OUTPUT_DIR))
    parser.add_argument(
        "--encoder_pretrain_path", type=str, default=None, help="Path to pretrained encoder"
    )
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=DETECTION_BATCH_SIZE)
    parser.add_argument("--num_epochs", type=int, default=DETECTION_NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=DETECTION_LR)
    parser.add_argument("--weight_decay", type=float, default=DETECTION_WEIGHT_DECAY)
    parser.add_argument("--num_queries", type=int, default=DETECTION_NUM_QUERIES)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--freeze_encoder_epochs",
        type=int,
        default=DETECTION_FREEZE_ENCODER_EPOCHS,
        help="Freeze encoder for first N epochs",
    )
    parser.add_argument("--save_every", type=int, default=DETECTION_SAVE_EVERY)

    args = parser.parse_args()

    train_detection(
        train_img_dir=args.train_img_dir,
        train_label_dir=args.train_label_dir,
        val_img_dir=args.val_img_dir,
        val_label_dir=args.val_label_dir,
        output_dir=args.output_dir,
        encoder_pretrain_path=args.encoder_pretrain_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_queries=args.num_queries,
        num_workers=args.num_workers,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
