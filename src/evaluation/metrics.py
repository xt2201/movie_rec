"""
Evaluation metrics for recommendation systems.

Implements standard ranking metrics:
- Precision@K
- Recall@K
- NDCG@K
- Hit Rate@K
- MAP@K
- MRR@K
- Coverage
"""
from __future__ import annotations

from typing import Union

import numpy as np
import torch


def _to_numpy(x: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    return x


def precision_at_k(
    predictions: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, set, torch.Tensor],
    k: int,
) -> float:
    """
    Compute Precision@K.
    
    Precision@K = |relevant ∩ recommended| / K
    
    Args:
        predictions: Ranked list of predicted items (shape: (k,) or more)
        ground_truth: Set/array of relevant items
        k: Number of recommendations to consider
        
    Returns:
        Precision@K score
    """
    predictions = _to_numpy(predictions)[:k]
    
    if isinstance(ground_truth, set):
        relevant = ground_truth
    else:
        relevant = set(_to_numpy(ground_truth).tolist())
    
    if len(relevant) == 0:
        return 0.0
    
    hits = sum(1 for item in predictions if item in relevant)
    return hits / k


def recall_at_k(
    predictions: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, set, torch.Tensor],
    k: int,
) -> float:
    """
    Compute Recall@K.
    
    Recall@K = |relevant ∩ recommended| / |relevant|
    
    Args:
        predictions: Ranked list of predicted items
        ground_truth: Set/array of relevant items
        k: Number of recommendations
        
    Returns:
        Recall@K score
    """
    predictions = _to_numpy(predictions)[:k]
    
    if isinstance(ground_truth, set):
        relevant = ground_truth
    else:
        relevant = set(_to_numpy(ground_truth).tolist())
    
    if len(relevant) == 0:
        return 0.0
    
    hits = sum(1 for item in predictions if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(
    predictions: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, set, torch.Tensor],
    k: int,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain @ K.
    
    NDCG@K = DCG@K / IDCG@K
    
    DCG@K = sum(rel_i / log2(i + 2)) for i in 0..k-1
    
    Args:
        predictions: Ranked list of predicted items
        ground_truth: Set/array of relevant items
        k: Number of recommendations
        
    Returns:
        NDCG@K score
    """
    predictions = _to_numpy(predictions)[:k]
    
    if isinstance(ground_truth, set):
        relevant = ground_truth
    else:
        relevant = set(_to_numpy(ground_truth).tolist())
    
    if len(relevant) == 0:
        return 0.0
    
    # Compute DCG
    dcg = 0.0
    for i, item in enumerate(predictions):
        if item in relevant:
            # Binary relevance
            dcg += 1.0 / np.log2(i + 2)
    
    # Compute IDCG (ideal DCG - all relevant items at top)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def hit_rate_at_k(
    predictions: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, set, torch.Tensor],
    k: int,
) -> float:
    """
    Compute Hit Rate @ K.
    
    Hit Rate@K = 1 if any relevant item in top-K, else 0
    
    Args:
        predictions: Ranked list of predicted items
        ground_truth: Set/array of relevant items
        k: Number of recommendations
        
    Returns:
        1.0 if hit, 0.0 otherwise
    """
    predictions = _to_numpy(predictions)[:k]
    
    if isinstance(ground_truth, set):
        relevant = ground_truth
    else:
        relevant = set(_to_numpy(ground_truth).tolist())
    
    for item in predictions:
        if item in relevant:
            return 1.0
    return 0.0


def map_at_k(
    predictions: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, set, torch.Tensor],
    k: int,
) -> float:
    """
    Compute Mean Average Precision @ K.
    
    AP@K = (1/min(m,K)) * sum(P@i * rel_i) for i in 1..K
    where m = number of relevant items
    
    Args:
        predictions: Ranked list of predicted items
        ground_truth: Set/array of relevant items
        k: Number of recommendations
        
    Returns:
        AP@K score (MAP@K when averaged over users)
    """
    predictions = _to_numpy(predictions)[:k]
    
    if isinstance(ground_truth, set):
        relevant = ground_truth
    else:
        relevant = set(_to_numpy(ground_truth).tolist())
    
    if len(relevant) == 0:
        return 0.0
    
    hits = 0
    sum_precision = 0.0
    
    for i, item in enumerate(predictions):
        if item in relevant:
            hits += 1
            sum_precision += hits / (i + 1)
    
    return sum_precision / min(len(relevant), k)


def mrr_at_k(
    predictions: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, set, torch.Tensor],
    k: int,
) -> float:
    """
    Compute Mean Reciprocal Rank @ K.
    
    RR@K = 1/rank of first relevant item (0 if none in top-K)
    
    Args:
        predictions: Ranked list of predicted items
        ground_truth: Set/array of relevant items
        k: Number of recommendations
        
    Returns:
        RR@K score (MRR@K when averaged over users)
    """
    predictions = _to_numpy(predictions)[:k]
    
    if isinstance(ground_truth, set):
        relevant = ground_truth
    else:
        relevant = set(_to_numpy(ground_truth).tolist())
    
    for i, item in enumerate(predictions):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def coverage(
    all_predictions: list[Union[np.ndarray, torch.Tensor]],
    num_items: int,
    k: int,
) -> float:
    """
    Compute catalog coverage.
    
    Coverage = |unique recommended items| / |total items|
    
    Args:
        all_predictions: List of predictions for all users
        num_items: Total number of items
        k: Number of recommendations per user
        
    Returns:
        Coverage ratio
    """
    recommended_items = set()
    
    for preds in all_predictions:
        preds = _to_numpy(preds)[:k]
        recommended_items.update(preds.tolist())
    
    return len(recommended_items) / num_items


def batch_precision_at_k(
    predictions: np.ndarray,
    ground_truth: dict[int, set[int]],
    k: int,
) -> float:
    """
    Compute average Precision@K for a batch of users.
    
    Args:
        predictions: Predictions array of shape (num_users, k)
        ground_truth: Dict mapping user_idx -> set of relevant items
        k: Number of recommendations
        
    Returns:
        Average Precision@K
    """
    scores = []
    for i, user_preds in enumerate(predictions):
        if i in ground_truth:
            score = precision_at_k(user_preds, ground_truth[i], k)
            scores.append(score)
    
    return np.mean(scores) if scores else 0.0


def batch_recall_at_k(
    predictions: np.ndarray,
    ground_truth: dict[int, set[int]],
    k: int,
) -> float:
    """Compute average Recall@K for a batch of users."""
    scores = []
    for i, user_preds in enumerate(predictions):
        if i in ground_truth:
            score = recall_at_k(user_preds, ground_truth[i], k)
            scores.append(score)
    
    return np.mean(scores) if scores else 0.0


def batch_ndcg_at_k(
    predictions: np.ndarray,
    ground_truth: dict[int, set[int]],
    k: int,
) -> float:
    """Compute average NDCG@K for a batch of users."""
    scores = []
    for i, user_preds in enumerate(predictions):
        if i in ground_truth:
            score = ndcg_at_k(user_preds, ground_truth[i], k)
            scores.append(score)
    
    return np.mean(scores) if scores else 0.0


def batch_hit_rate_at_k(
    predictions: np.ndarray,
    ground_truth: dict[int, set[int]],
    k: int,
) -> float:
    """Compute average Hit Rate@K for a batch of users."""
    scores = []
    for i, user_preds in enumerate(predictions):
        if i in ground_truth:
            score = hit_rate_at_k(user_preds, ground_truth[i], k)
            scores.append(score)
    
    return np.mean(scores) if scores else 0.0
