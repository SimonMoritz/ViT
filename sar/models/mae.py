"""Masked Autoencoder (MAE) for self-supervised pretraining."""

import torch
import torch.nn as nn
import numpy as np


class MAEDecoder(nn.Module):
    """Lightweight decoder for MAE reconstruction."""

    def __init__(
        self,
        embed_dim=192,
        decoder_embed_dim=128,
        decoder_depth=4,
        decoder_n_heads=4,
        patch_size=16,
        in_channels=3,
    ):
        super().__init__()
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size
        self.in_channels = in_channels

        # Projection from encoder dim to decoder dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Decoder position embeddings (will be initialized based on input size)
        self.decoder_pos_embed = None

        # Transformer decoder blocks
        from sar.models.vit import TransformerBlock
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=decoder_embed_dim,
                n_heads=decoder_n_heads,
                mlp_ratio=4.0,
                dropout=0.0
            )
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Reconstruct pixels
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size ** 2 * in_channels
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _init_pos_embed(self, n_patches, device):
        """Initialize positional embeddings if not already done."""
        if self.decoder_pos_embed is None or self.decoder_pos_embed.shape[1] != n_patches:
            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, n_patches, self.decoder_embed_dim, device=device),
                requires_grad=True
            )
            nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def forward(self, x, mask_indices):
        """
        Args:
            x: (B, N_visible, D) - visible patch tokens from encoder
            mask_indices: (B, N_total) - boolean mask, True = masked
        Returns:
            (B, N_total, patch_size^2 * C) - reconstructed patches
        """
        B, N_visible, D = x.shape
        N_total = mask_indices.shape[1]
        device = x.device

        # Initialize positional embeddings on correct device
        self._init_pos_embed(N_total, device)

        # Project to decoder dimension
        x = self.decoder_embed(x)  # (B, N_visible, decoder_D)

        # Create full sequence with mask tokens
        mask_tokens = self.mask_token.repeat(B, N_total - N_visible, 1)  # (B, N_masked, decoder_D)

        # Combine visible and masked tokens
        # We need to unshuffle: put visible tokens back in their original positions
        x_full = torch.zeros(B, N_total, self.decoder_embed_dim, device=device)

        # Place visible tokens in correct positions
        visible_indices = ~mask_indices  # (B, N_total)
        for i in range(B):
            visible_pos = visible_indices[i].nonzero(as_tuple=True)[0]
            x_full[i, visible_pos] = x[i]

        # Place mask tokens in masked positions
        for i in range(B):
            masked_pos = mask_indices[i].nonzero(as_tuple=True)[0]
            x_full[i, masked_pos] = mask_tokens[i, :len(masked_pos)]

        # Add positional embeddings
        x_full = x_full + self.decoder_pos_embed

        # Decoder blocks
        for block in self.decoder_blocks:
            x_full = block(x_full)

        x_full = self.decoder_norm(x_full)

        # Predict pixel values
        x_full = self.decoder_pred(x_full)  # (B, N_total, patch_size^2 * C)

        return x_full


class MAE(nn.Module):
    """
    Masked Autoencoder (MAE) for self-supervised pretraining.

    Following "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021)
    """

    def __init__(
        self,
        encoder,
        mask_ratio=0.75,
        decoder_embed_dim=128,
        decoder_depth=4,
        decoder_n_heads=4,
        patch_size=16,
        in_channels=3,
    ):
        """
        Args:
            encoder: ViT encoder (e.g., ViTTiny)
            mask_ratio: Ratio of patches to mask (default: 0.75 as in MAE paper)
            decoder_embed_dim: Decoder embedding dimension
            decoder_depth: Number of decoder transformer blocks
            decoder_n_heads: Number of attention heads in decoder
            patch_size: Size of patches
            in_channels: Number of input channels
        """
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.in_channels = in_channels

        # MAE decoder
        self.decoder = MAEDecoder(
            embed_dim=encoder.embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_n_heads=decoder_n_heads,
            patch_size=patch_size,
            in_channels=in_channels,
        )

    def patchify(self, imgs):
        """
        Convert images to patches.

        Args:
            imgs: (B, C, H, W)
        Returns:
            (B, N, patch_size^2 * C)
        """
        B, C, H, W = imgs.shape
        p = self.patch_size
        h, w = H // p, W // p

        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, h, w, p, p, C)
        x = x.reshape(B, h * w, p * p * C)  # (B, N, patch_size^2 * C)

        return x

    def unpatchify(self, x, H, W):
        """
        Convert patches back to images.

        Args:
            x: (B, N, patch_size^2 * C)
            H, W: Original image height and width
        Returns:
            (B, C, H, W)
        """
        B = x.shape[0]
        p = self.patch_size
        C = self.in_channels
        h, w = H // p, W // p

        x = x.reshape(B, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
        imgs = x.reshape(B, C, H, W)

        return imgs

    def random_masking(self, x):
        """
        Perform random masking by shuffling patches.

        Args:
            x: (B, N, D) - patch tokens
        Returns:
            x_masked: (B, N_visible, D) - visible patch tokens
            mask: (B, N) - boolean mask, True = masked
            ids_restore: (B, N) - indices to restore original order
        """
        B, N, D = x.shape
        n_keep = int(N * (1 - self.mask_ratio))

        # Random shuffle
        noise = torch.rand(B, N, device=x.device)  # Uniform [0, 1)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first n_keep patches
        ids_keep = ids_shuffle[:, :n_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        # Generate mask: 0 = keep, 1 = remove
        mask = torch.ones([B, N], device=x.device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)

        return x_masked, mask, ids_restore

    def forward(self, imgs):
        """
        Forward pass for MAE pretraining.

        Args:
            imgs: (B, C, H, W)
        Returns:
            loss: Reconstruction loss
            pred: Reconstructed images (B, C, H, W)
            mask: Binary mask (B, N)
        """
        B, C, H, W = imgs.shape

        # Encode with masking
        # Get patch embeddings (before transformer)
        x = self.encoder.patch_embed(imgs)  # (B, N, D)

        # Add positional embeddings (no CLS token for MAE)
        if self.encoder.use_cls_token:
            # Skip CLS token position embedding
            x = x + self.encoder.pos_embed[:, 1:, :]
        else:
            x = x + self.encoder.pos_embed

        # Random masking
        x_masked, mask, ids_restore = self.random_masking(x)

        # Apply transformer blocks to visible patches
        for block in self.encoder.blocks:
            x_masked = block(x_masked)

        x_masked = self.encoder.norm(x_masked)

        # Decode
        pred_patches = self.decoder(x_masked, mask)  # (B, N, patch_size^2 * C)

        # Compute reconstruction loss (only on masked patches)
        target = self.patchify(imgs)  # (B, N, patch_size^2 * C)

        # Normalize targets per patch (improves representation quality)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6) ** 0.5

        # MSE loss only on masked patches
        loss = (pred_patches - target) ** 2
        loss = loss.mean(dim=-1)  # (B, N) - mean loss per patch

        # Only compute loss on masked patches
        loss = (loss * mask).sum() / mask.sum()  # Mean loss on masked patches

        # Reconstruct full image for visualization
        with torch.no_grad():
            pred_patches_full = pred_patches.detach()
            # De-normalize
            pred_patches_full = pred_patches_full * (var + 1e-6) ** 0.5 + mean
            pred_imgs = self.unpatchify(pred_patches_full, H, W)

        return loss, pred_imgs, mask


if __name__ == "__main__":
    from sar.models.vit import ViTTiny

    # Test MAE
    encoder = ViTTiny(img_size=224, use_cls_token=False)
    mae = MAE(encoder, mask_ratio=0.75, patch_size=16)

    x = torch.randn(2, 3, 224, 224)
    loss, pred, mask = mae(x)

    print(f"Input shape: {x.shape}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Masked ratio: {mask.float().mean():.2f}")
    print(f"MAE params: {sum(p.numel() for p in mae.parameters()) / 1e6:.2f}M")
