"""Vision Transformer (ViT-Tiny) implementation."""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Convolutional projection
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, N, D) where N = number of patches, D = embed_dim
        """
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2)  # (B, D, N)
        x = x.transpose(1, 2)  # (B, N, D)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, embed_dim=192, n_heads=3, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, N, D)
        Returns:
            (B, N, D)
        """
        B, N, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D_h)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v)  # (B, H, N, D_h)
        x = x.transpose(1, 2)  # (B, N, H, D_h)
        x = x.reshape(B, N, D)  # (B, N, D)

        # Output projection
        x = self.proj(x)
        x = self.dropout(x)

        return x


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, embed_dim=192, hidden_dim=768, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, embed_dim=192, n_heads=3, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTiny(nn.Module):
    """
    Vision Transformer - Tiny variant.

    Architecture:
    - Patch size: 16x16
    - Embedding dim: 192
    - Depth: 12 layers
    - Heads: 3
    - MLP ratio: 4
    - Params: ~5M
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=192,
        depth=12,
        n_heads=3,
        mlp_ratio=4.0,
        dropout=0.0,
        use_cls_token=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        n_patches = self.patch_embed.n_patches

        # CLS token (optional, useful for classification tasks)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            n_tokens = n_patches + 1
        else:
            self.cls_token = None
            n_tokens = n_patches

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following ViT paper."""
        # Position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Apply weight initialization to all modules
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        """Initialize individual module weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # Kaiming initialization for conv layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input images
        Returns:
            (B, N, D) sequence of patch tokens
            If use_cls_token=True, N = n_patches + 1
            If use_cls_token=False, N = n_patches
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, N_patches, D)

        # Add CLS token if used
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, N_patches+1, D)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        return x

    def get_patch_tokens(self, x):
        """
        Get patch tokens without CLS token.
        Useful for dense prediction tasks like object detection.

        Args:
            x: (B, C, H, W)
        Returns:
            (B, N_patches, D)
        """
        tokens = self.forward(x)
        if self.use_cls_token:
            return tokens[:, 1:, :]  # Remove CLS token
        return tokens

    def get_cls_token(self, x):
        """
        Get only the CLS token.
        Useful for classification tasks.

        Args:
            x: (B, C, H, W)
        Returns:
            (B, D)
        """
        assert self.use_cls_token, "CLS token not enabled"
        tokens = self.forward(x)
        return tokens[:, 0, :]  # First token is CLS


if __name__ == "__main__":
    # Test the model
    model = ViTTiny(img_size=224, use_cls_token=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
