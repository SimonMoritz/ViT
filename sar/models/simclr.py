"""SimCLR for contrastive self-supervised pretraining."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Projection head for SimCLR.
    Maps representations to a space where contrastive loss is applied.
    """

    def __init__(self, input_dim=192, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    """
    SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.

    Based on "A Simple Framework for Contrastive Learning of Visual Representations"
    (Chen et al., 2020)
    """

    def __init__(
        self,
        encoder,
        projection_dim=128,
        projection_hidden_dim=512,
        temperature=0.5,
    ):
        """
        Args:
            encoder: ViT encoder (e.g., ViTTiny)
            projection_dim: Output dimension of projection head
            projection_hidden_dim: Hidden dimension of projection head
            temperature: Temperature parameter for contrastive loss
        """
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

        # Projection head
        self.projection_head = ProjectionHead(
            input_dim=encoder.embed_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim,
        )

    def forward(self, x1, x2):
        """
        Forward pass for SimCLR training.

        Args:
            x1: (B, C, H, W) - first augmented view
            x2: (B, C, H, W) - second augmented view
        Returns:
            loss: NT-Xent contrastive loss
            z1: Projected representations for x1
            z2: Projected representations for x2
        """
        B = x1.shape[0]

        # Encode both views
        h1 = self.encode(x1)  # (B, D)
        h2 = self.encode(x2)  # (B, D)

        # Project to contrastive space
        z1 = self.projection_head(h1)  # (B, projection_dim)
        z2 = self.projection_head(h2)  # (B, projection_dim)

        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Compute NT-Xent loss
        loss = self.nt_xent_loss(z1, z2)

        return loss, z1, z2

    def encode(self, x):
        """
        Encode image to representation.

        Args:
            x: (B, C, H, W)
        Returns:
            (B, D) - representation vector
        """
        tokens = self.encoder(x)  # (B, N, D)

        # Global average pooling over all patch tokens
        if self.encoder.use_cls_token:
            # Use CLS token
            h = tokens[:, 0, :]
        else:
            # Average pool over all patches
            h = tokens.mean(dim=1)

        return h

    def nt_xent_loss(self, z1, z2):
        """
        Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).

        Args:
            z1: (B, D) - normalized projections from view 1
            z2: (B, D) - normalized projections from view 2
        Returns:
            loss: Scalar contrastive loss
        """
        B = z1.shape[0]
        device = z1.device

        # Concatenate z1 and z2 to form 2B samples
        z = torch.cat([z1, z2], dim=0)  # (2B, D)

        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # Create mask to exclude self-similarity
        mask = torch.eye(2 * B, device=device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        # Positive pairs: (i, i+B) and (i+B, i)
        # For each sample i in first half, positive is i+B
        # For each sample i in second half, positive is i-B

        # Labels for first half (0 to B-1)
        # Positive is at index i+B in the concatenated similarity row
        # After removing diagonal, positive is at position i+B-1 if i+B > i, else i+B
        # Simpler: create explicit positive mask

        # Create positive pair labels
        # For sample i in [0, B), positive is B+i
        # For sample i in [B, 2B), positive is i-B
        labels = torch.zeros(2 * B, device=device, dtype=torch.long)
        labels[:B] = torch.arange(B, 2 * B, device=device)
        labels[B:] = torch.arange(0, B, device=device)

        # Adjust labels for removed diagonal
        # Since we masked diagonal with -inf, we need to adjust indices
        # after diagonal is removed conceptually. However, with masking,
        # we can use cross_entropy directly on the similarity matrix.

        # Actually, we need to be more careful here.
        # Standard approach: for each anchor, compute softmax over all negatives + 1 positive
        # Cross-entropy loss with the positive pair

        # Let's compute manually for clarity
        # For sample i, compute: -log( exp(sim(i, positive)) / sum_j exp(sim(i, j)) )
        # where j != i

        losses = []
        for i in range(2 * B):
            # Positive sample
            pos_idx = labels[i]

            # Similarity with positive
            pos_sim = sim_matrix[i, pos_idx]

            # All similarities except self
            # Create a mask for the current sample's row
            row_mask = torch.ones(2 * B, device=device, dtype=torch.bool)
            row_mask[i] = False

            # Get all similarities except self
            all_sims = sim_matrix[i, row_mask]

            # Denominator: exp(positive) + sum of exp(all negatives)
            # But positive is also in all_sims, so we can use logsumexp directly
            # Actually, all_sims contains positive + negatives (excluding self)

            # Numerator: exp(pos_sim)
            # Denominator: sum of exp(all_sims)
            # Loss: -log(exp(pos_sim) / sum(exp(all_sims)))
            #     = -pos_sim + log(sum(exp(all_sims)))
            #     = -pos_sim + logsumexp(all_sims)

            loss_i = -pos_sim + torch.logsumexp(all_sims, dim=0)
            losses.append(loss_i)

        loss = torch.stack(losses).mean()
        return loss

    def get_encoder(self):
        """Return the encoder (useful after pretraining)."""
        return self.encoder


if __name__ == "__main__":
    from sar.models.vit import ViTTiny

    # Test SimCLR
    encoder = ViTTiny(img_size=224, use_cls_token=False)
    simclr = SimCLR(encoder, projection_dim=128, temperature=0.5)

    # Two augmented views of the same batch
    x1 = torch.randn(4, 3, 224, 224)
    x2 = torch.randn(4, 3, 224, 224)

    loss, z1, z2 = simclr(x1, x2)

    print(f"Input shape: {x1.shape}")
    print(f"Projection z1 shape: {z1.shape}")
    print(f"Projection z2 shape: {z2.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"SimCLR params: {sum(p.numel() for p in simclr.parameters()) / 1e6:.2f}M")
