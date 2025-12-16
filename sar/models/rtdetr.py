"""RT-DETR: Real-Time Detection Transformer head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP(nn.Module):
    """Simple MLP."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer for DETR."""

    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Feed-forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt: (B, num_queries, d_model) - query embeddings
            memory: (B, H*W, d_model) - encoder output
            tgt_mask: Optional mask for self-attention
            memory_mask: Optional mask for cross-attention
        Returns:
            (B, num_queries, d_model)
        """
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention with encoder output
        tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed-forward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class RTDETR(nn.Module):
    """
    RT-DETR: Real-Time Detection Transformer.

    Simplified version for single-class airport detection.
    """

    def __init__(
        self,
        encoder,
        num_classes=1,  # Single class: airport
        num_queries=100,
        hidden_dim=256,
        nheads=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        """
        Args:
            encoder: ViT encoder
            num_classes: Number of object classes (1 for airport)
            num_queries: Number of object queries
            hidden_dim: Hidden dimension for transformer decoder
            nheads: Number of attention heads
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Project encoder output to decoder hidden dimension
        self.input_proj = nn.Linear(encoder.embed_dim, hidden_dim)

        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Transformer decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, nheads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # Predict 4 bbox coords

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass for detection.

        Args:
            x: (B, C, H, W) - input images
        Returns:
            pred_logits: (B, num_queries, num_classes+1) - class predictions
            pred_boxes: (B, num_queries, 4) - bbox predictions (cx, cy, w, h) normalized [0, 1]
        """
        B = x.shape[0]

        # Encode image
        memory = self.encoder(x)  # (B, N_patches, D)

        # Remove CLS token if present
        if self.encoder.use_cls_token:
            memory = memory[:, 1:, :]

        # Project to decoder hidden dimension
        memory = self.input_proj(memory)  # (B, N_patches, hidden_dim)

        # Get query embeddings
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, num_queries, hidden_dim)

        # Initialize queries (could also use learned initialization)
        tgt = torch.zeros_like(query_embed)

        # Decoder
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory)

        # Predictions
        pred_logits = self.class_embed(tgt)  # (B, num_queries, num_classes+1)
        pred_boxes = self.bbox_embed(tgt).sigmoid()  # (B, num_queries, 4), normalized to [0, 1]

        return pred_logits, pred_boxes

    def get_encoder(self):
        """Return the encoder."""
        return self.encoder


class RTDETRLoss(nn.Module):
    """
    Loss function for RT-DETR.

    Uses Hungarian matching to assign predictions to ground truth,
    then computes classification and box regression losses.
    """

    def __init__(self, num_classes=1, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        """
        Args:
            num_classes: Number of classes (excluding background)
            cost_class: Weight for classification cost in matching
            cost_bbox: Weight for L1 bbox cost in matching
            cost_giou: Weight for GIoU cost in matching
        """
        super().__init__()
        self.num_classes = num_classes
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def forward(self, pred_logits, pred_boxes, targets):
        """
        Args:
            pred_logits: (B, num_queries, num_classes+1)
            pred_boxes: (B, num_queries, 4) - cx, cy, w, h normalized
            targets: List of dicts, length B, each containing:
                - 'labels': (N,) class labels
                - 'boxes': (N, 4) bounding boxes cx, cy, w, h normalized
        Returns:
            loss_dict: Dictionary with loss components
        """
        B, num_queries, _ = pred_logits.shape
        device = pred_logits.device

        # Flatten batch dimension for matching
        pred_logits_flat = pred_logits.flatten(0, 1)  # (B*num_queries, num_classes+1)
        pred_boxes_flat = pred_boxes.flatten(0, 1)  # (B*num_queries, 4)

        # Concatenate all targets
        target_labels = []
        target_boxes = []
        batch_idx = []
        for i, t in enumerate(targets):
            target_labels.append(t['labels'])
            target_boxes.append(t['boxes'])
            batch_idx.extend([i] * len(t['labels']))

        if len(target_labels) == 0:
            # No targets in batch
            losses = {
                'loss_ce': pred_logits.sum() * 0.0,  # Dummy loss
                'loss_bbox': pred_boxes.sum() * 0.0,
                'loss_giou': pred_boxes.sum() * 0.0,
            }
            losses['loss'] = sum(losses.values())
            return losses

        target_labels = torch.cat(target_labels).to(device)
        target_boxes = torch.cat(target_boxes).to(device)
        batch_idx = torch.tensor(batch_idx, device=device)

        # Hungarian matching
        indices = self.hungarian_matching(
            pred_logits, pred_boxes, targets
        )

        # Classification loss
        # Create target classes for all queries (most are background)
        target_classes_o = torch.full(
            (B, num_queries), self.num_classes, dtype=torch.long, device=device
        )  # Background class

        # Assign matched targets
        for batch_i, (src_idx, tgt_idx) in enumerate(indices):
            target_classes_o[batch_i, src_idx] = targets[batch_i]['labels'][tgt_idx]

        # Classification loss (cross-entropy)
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2), target_classes_o, weight=None
        )

        # Bbox losses (only for matched predictions)
        idx_src = []
        idx_tgt = []
        for batch_i, (src_idx, tgt_idx) in enumerate(indices):
            idx_src.append(src_idx + batch_i * num_queries)
            idx_tgt.append(tgt_idx + sum(len(t['labels']) for t in targets[:batch_i]))

        if len(idx_src) > 0:
            idx_src = torch.cat(idx_src)
            idx_tgt = torch.cat(idx_tgt)

            # Select matched predictions and targets
            pred_boxes_matched = pred_boxes_flat[idx_src]
            target_boxes_matched = torch.cat([t['boxes'] for t in targets])[idx_tgt]

            # L1 loss
            loss_bbox = F.l1_loss(pred_boxes_matched, target_boxes_matched, reduction='mean')

            # GIoU loss
            loss_giou = self.giou_loss(pred_boxes_matched, target_boxes_matched)
        else:
            loss_bbox = pred_boxes.sum() * 0.0
            loss_giou = pred_boxes.sum() * 0.0

        losses = {
            'loss_ce': loss_ce,
            'loss_bbox': loss_bbox * 5.0,  # Weight bbox loss
            'loss_giou': loss_giou * 2.0,  # Weight GIoU loss
        }
        losses['loss'] = sum(losses.values())

        return losses

    def hungarian_matching(self, pred_logits, pred_boxes, targets):
        """
        Perform Hungarian matching between predictions and targets.

        Args:
            pred_logits: (B, num_queries, num_classes+1)
            pred_boxes: (B, num_queries, 4)
            targets: List of target dicts

        Returns:
            List of tuples (src_idx, tgt_idx) for each batch
        """
        from scipy.optimize import linear_sum_assignment

        B, num_queries = pred_logits.shape[:2]
        indices = []

        for batch_i in range(B):
            # Get predictions for this batch
            out_prob = pred_logits[batch_i].softmax(-1)  # (num_queries, num_classes+1)
            out_bbox = pred_boxes[batch_i]  # (num_queries, 4)

            # Get targets for this batch
            tgt_ids = targets[batch_i]['labels']  # (num_targets,)
            tgt_bbox = targets[batch_i]['boxes']  # (num_targets, 4)

            if len(tgt_ids) == 0:
                # No targets, no matching
                indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                continue

            # Classification cost
            cost_class = -out_prob[:, tgt_ids]  # (num_queries, num_targets)

            # Bbox L1 cost
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # (num_queries, num_targets)

            # GIoU cost
            cost_giou = -self.generalized_box_iou(out_bbox, tgt_bbox)  # (num_queries, num_targets)

            # Final cost matrix
            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            C = C.cpu().detach().numpy()

            # Hungarian algorithm
            src_idx, tgt_idx = linear_sum_assignment(C)

            indices.append((torch.as_tensor(src_idx, dtype=torch.long), torch.as_tensor(tgt_idx, dtype=torch.long)))

        return indices

    def generalized_box_iou(self, boxes1, boxes2):
        """
        Compute generalized IoU between two sets of boxes.
        Boxes are in (cx, cy, w, h) format, normalized.

        Args:
            boxes1: (N, 4)
            boxes2: (M, 4)
        Returns:
            (N, M) GIoU matrix
        """
        # Convert to (x1, y1, x2, y2)
        boxes1_xyxy = self.box_cxcywh_to_xyxy(boxes1)
        boxes2_xyxy = self.box_cxcywh_to_xyxy(boxes2)

        # Compute IoU
        iou = self.box_iou(boxes1_xyxy, boxes2_xyxy)

        # Compute enclosing box
        lt = torch.min(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
        rb = torch.max(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        area_c = wh[:, :, 0] * wh[:, :, 1]

        # GIoU
        area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
        area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
        union = area1[:, None] + area2[None, :] - iou * area1[:, None]

        giou = iou - (area_c - union) / area_c

        return giou

    def box_iou(self, boxes1, boxes2):
        """
        Compute IoU between two sets of boxes (x1, y1, x2, y2).

        Args:
            boxes1: (N, 4)
            boxes2: (M, 4)
        Returns:
            (N, M) IoU matrix
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2[None, :] - inter

        iou = inter / union.clamp(min=1e-6)
        return iou

    def box_cxcywh_to_xyxy(self, boxes):
        """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)."""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def giou_loss(self, pred_boxes, target_boxes):
        """Compute GIoU loss for matched boxes."""
        giou = torch.diagonal(self.generalized_box_iou(pred_boxes, target_boxes))
        loss = 1 - giou
        return loss.mean()


if __name__ == "__main__":
    from sar.models.vit import ViTTiny

    # Test RT-DETR
    encoder = ViTTiny(img_size=224, use_cls_token=False)
    rtdetr = RTDETR(encoder, num_classes=1, num_queries=100)

    x = torch.randn(2, 3, 224, 224)
    pred_logits, pred_boxes = rtdetr(x)

    print(f"Input shape: {x.shape}")
    print(f"Pred logits shape: {pred_logits.shape}")
    print(f"Pred boxes shape: {pred_boxes.shape}")
    print(f"RT-DETR params: {sum(p.numel() for p in rtdetr.parameters()) / 1e6:.2f}M")

    # Test loss
    targets = [
        {'labels': torch.tensor([0, 0]), 'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.1, 0.2]])},
        {'labels': torch.tensor([0]), 'boxes': torch.tensor([[0.6, 0.6, 0.15, 0.25]])},
    ]

    criterion = RTDETRLoss(num_classes=1)
    losses = criterion(pred_logits, pred_boxes, targets)
    print(f"\nLosses: {losses}")
