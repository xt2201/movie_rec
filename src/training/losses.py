"""
Loss functions for recommendation systems.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss.
    
    BPR optimizes pairwise ranking: positive items should be ranked
    higher than negative items for each user.
    
    L_BPR = -mean(log(sigmoid(pos_score - neg_score)))
    """
    
    def __init__(self, reduction: str = "mean"):
        """
        Initialize BPR loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BPR loss.
        
        Args:
            pos_scores: Scores for positive items (batch_size,)
            neg_scores: Scores for negative items (batch_size,)
            
        Returns:
            BPR loss
        """
        loss = -F.logsigmoid(pos_scores - neg_scores)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class BCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss for implicit feedback.
    
    Treats recommendation as binary classification:
    positive interactions = 1, negative = 0.
    """
    
    def __init__(self, reduction: str = "mean"):
        """
        Initialize BCE loss.
        
        Args:
            reduction: Reduction method
        """
        super().__init__()
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward(
        self,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BCE loss.
        
        Args:
            pos_logits: Logits for positive items
            neg_logits: Logits for negative items
            
        Returns:
            BCE loss
        """
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([pos_labels, neg_labels])
        
        return self.criterion(logits, labels)


class MarginLoss(nn.Module):
    """
    Margin-based ranking loss.
    
    L = max(0, margin - (pos_score - neg_score))
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        loss = torch.clamp(self.margin - (pos_scores - neg_scores), min=0)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_bpr_loss(
    user_emb: torch.Tensor,
    pos_item_emb: torch.Tensor,
    neg_item_emb: torch.Tensor,
    reg_weight: float = 0.0,
    initial_user_emb: torch.Tensor | None = None,
    initial_pos_emb: torch.Tensor | None = None,
    initial_neg_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute BPR loss with optional L2 regularization on initial embeddings.
    
    This is the standard loss function for LightGCN and similar models.
    
    Args:
        user_emb: User embeddings from GCN (batch_size, dim)
        pos_item_emb: Positive item embeddings (batch_size, dim)
        neg_item_emb: Negative item embeddings (batch_size, dim)
        reg_weight: L2 regularization weight
        initial_user_emb: Initial user embeddings for regularization
        initial_pos_emb: Initial positive item embeddings
        initial_neg_emb: Initial negative item embeddings
        
    Returns:
        Total loss
    """
    # Compute scores via dot product
    pos_scores = (user_emb * pos_item_emb).sum(dim=1)
    neg_scores = (user_emb * neg_item_emb).sum(dim=1)
    
    # BPR loss
    bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    
    # L2 regularization on initial embeddings (not GCN outputs)
    if reg_weight > 0 and initial_user_emb is not None:
        reg_loss = reg_weight * (
            initial_user_emb.norm(2).pow(2) +
            initial_pos_emb.norm(2).pow(2) +
            initial_neg_emb.norm(2).pow(2)
        ) / len(user_emb)
        return bpr_loss + reg_loss
    
    return bpr_loss


def compute_l2_reg(
    *embeddings: torch.Tensor,
    weight: float = 1e-5,
) -> torch.Tensor:
    """
    Compute L2 regularization loss.
    
    Args:
        *embeddings: Embedding tensors to regularize
        weight: Regularization weight
        
    Returns:
        L2 regularization loss
    """
    reg = sum(emb.norm(2).pow(2) for emb in embeddings)
    return weight * reg
