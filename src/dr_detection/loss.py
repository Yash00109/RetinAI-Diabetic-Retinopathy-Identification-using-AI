from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with optional class weights and label smoothing.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Effectively addresses severe medical class imbalance by down-weighting the loss
    assigned to easy, well-classified examples (e.g. obvious No DR) and focusing
    gradients on ambiguous and rare pathology (e.g. Mild and Severe DR).
    """
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross entropy with label smoothing
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        
        # p_t: probability of true class
        pt = torch.exp(-ce_loss)
        
        # Focal weight: (1 - p_t)^gamma
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class OrdinalDistanceLoss(nn.Module):
    """Combines Cross Entropy with an Ordinal Quadratic Distance Penalty:
    L = L_CE + lambda * sum_j P(pred=j) * (j - true_label)^2
    
    Directly aligns neural network training with the Quadratic Weighted Kappa (QWK)
    evaluation metric by penalizing multi-grade misclassifications proportionally
    to the squared clinical distance.
    """
    def __init__(
        self,
        num_classes: int = 5,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.05,
        distance_weight: float = 0.35,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = label_smoothing
        self.distance_weight = distance_weight
        
        # Precompute class grade index grid: [0, 1, 2, 3, 4]
        self.register_buffer("grades", torch.arange(num_classes, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )
        
        probs = F.softmax(logits, dim=1) # [B, num_classes]
        # Expected score: sum_j j * p_j
        expected_score = torch.sum(probs * self.grades, dim=1) # [B]
        
        # Smooth L1 / MSE distance penalty
        distance_penalty = F.smooth_l1_loss(expected_score, targets.float(), beta=1.0)
        
        return ce + self.distance_weight * distance_penalty
