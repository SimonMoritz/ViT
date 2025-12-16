"""Training script for MAE pretraining."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import argparse

from sar.models.vit import ViTTiny
from sar.models.mae import MAE
from sar.data.datasets import PretrainDataset
from sar.augmentation import get_pretrain_augmentation


def train_mae(
    img_dir="Airport_Dataset_v0_images",
    output_dir="checkpoints/mae",
    img_size=224,
    batch_size=32,
    num_epochs=300,
    lr=1.5e-4,
    weight_decay=0.05,
    mask_ratio=0.75,
    num_workers=4,
    save_every=50,
):
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(output_dir / "logs")

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
    if img_size >= 512:
        decoder_config = {
            'decoder_embed_dim': 256,  # 2x larger for high-res
            'decoder_depth': 8,        # 2x deeper
            'decoder_n_heads': 8,      # 2x more heads
        }
        print(f"Using LARGE decoder for {img_size}px: dim=256, depth=8, heads=8")
    else:
        decoder_config = {
            'decoder_embed_dim': 128,
            'decoder_depth': 4,
            'decoder_n_heads': 4,
        }
        print(f"Using standard decoder for {img_size}px: dim=128, depth=4, heads=4")

    mae = MAE(encoder, mask_ratio=mask_ratio, patch_size=16, **decoder_config)
    mae = mae.to(device)

    print(f"Model parameters: {sum(p.numel() for p in mae.parameters()) / 1e6:.2f}M")

    # Optimizer (AdamW with cosine schedule as in MAE paper)
    optimizer = torch.optim.AdamW(
        mae.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay
    )

    # Cosine learning rate schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=lr * 0.01
    )

    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        mae.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, images in enumerate(pbar):
            images = images.to(device)

            # Forward pass
            loss, pred_imgs, mask = mae(images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            epoch_loss += loss.item()
            global_step += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

            # TensorBoard logging
            if global_step % 10 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

            # Log reconstructions
            if global_step % 100 == 0:
                with torch.no_grad():
                    # Visualize first image in batch
                    original = images[0].cpu()
                    reconstructed = pred_imgs[0].cpu()
                    mask_vis = mask[0].cpu()

                    writer.add_image('mae/original', original, global_step)
                    writer.add_image('mae/reconstructed', reconstructed.clamp(0, 1), global_step)

        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)

        # Learning rate schedule step
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            checkpoint_path = output_dir / f"mae_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': mae.state_dict(),
                'encoder_state_dict': mae.encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final encoder
    encoder_path = output_dir / "encoder_mae_final.pth"
    torch.save(mae.encoder.state_dict(), encoder_path)
    print(f"Saved final encoder: {encoder_path}")

    writer.close()
    print("MAE training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train MAE for self-supervised pretraining')
    parser.add_argument('--img_dir', type=str, default='Airport_Dataset_v0_images',
                        help='Directory with images')
    parser.add_argument('--output_dir', type=str, default='checkpoints/mae',
                        help='Output directory for checkpoints')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1.5e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                        help='Mask ratio')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save checkpoint every N epochs')

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
