"""Training script for SimCLR pretraining."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import argparse

from sar.models.vit import ViTTiny
from sar.models.simclr import SimCLR
from sar.data.datasets import PretrainDataset
from sar.augmentation import get_simclr_augmentation, DualViewTransform


def train_simclr(
    img_dir="Airport_Dataset_v0_images",
    output_dir="checkpoints/simclr",
    encoder_pretrain_path=None,  # Optional: load MAE pretrained encoder
    img_size=224,
    batch_size=32,
    num_epochs=200,
    lr=3e-4,
    weight_decay=1e-4,
    temperature=0.5,
    num_workers=4,
    save_every=50,
):
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(output_dir / "logs")

    # Dataset and dataloader (dual view transform)
    transform = DualViewTransform(get_simclr_augmentation(img_size))
    dataset = PretrainDataset(img_dir, transform=transform)

    # Custom collate function for dual views
    def collate_fn(batch):
        view1 = torch.stack([item[0] for item in batch])
        view2 = torch.stack([item[1] for item in batch])
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
        state_dict = torch.load(encoder_pretrain_path, map_location='cpu')
        encoder.load_state_dict(state_dict)

    simclr = SimCLR(encoder, projection_dim=128, temperature=temperature)
    simclr = simclr.to(device)

    print(f"Model parameters: {sum(p.numel() for p in simclr.parameters()) / 1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        simclr.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
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
        simclr.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (view1, view2) in enumerate(pbar):
            view1 = view1.to(device)
            view2 = view2.to(device)

            # Forward pass
            loss, z1, z2 = simclr(view1, view2)

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

            # Log sample views
            if global_step % 100 == 0:
                writer.add_image('simclr/view1', view1[0].cpu(), global_step)
                writer.add_image('simclr/view2', view2[0].cpu(), global_step)

        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)

        # Learning rate schedule step
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            checkpoint_path = output_dir / f"simclr_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': simclr.state_dict(),
                'encoder_state_dict': simclr.encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final encoder
    encoder_path = output_dir / "encoder_simclr_final.pth"
    torch.save(simclr.encoder.state_dict(), encoder_path)
    print(f"Saved final encoder: {encoder_path}")

    writer.close()
    print("SimCLR training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train SimCLR for contrastive learning')
    parser.add_argument('--img_dir', type=str, default='Airport_Dataset_v0_images',
                        help='Directory with images')
    parser.add_argument('--output_dir', type=str, default='checkpoints/simclr',
                        help='Output directory for checkpoints')
    parser.add_argument('--encoder_pretrain_path', type=str, default=None,
                        help='Path to MAE pretrained encoder (optional)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for contrastive loss')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save checkpoint every N epochs')

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
