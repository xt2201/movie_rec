"""
Sampled Negative Evaluator for Leave-One-Out evaluation.

This implements the standard paper evaluation protocol:
- For each test user, sample N random negatives
- Rank the 1 positive item among the N negatives
- Compute metrics on this small ranking
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm


def sampled_evaluate(
    model,
    ground_truth: dict[int, set[int]],
    train_items: dict[int, set[int]],
    num_items: int,
    num_neg_samples: int = 99,
    k_values: list[int] = [5, 10, 20],
    edge_index: torch.Tensor = None,
    edge_weight: torch.Tensor = None,
    device: torch.device = None,
) -> dict[str, float]:
    """
    Evaluate using sampled negative protocol (standard for LOO).
    
    For each user:
    1. Get their positive test item(s)
    2. Sample num_neg_samples random negative items
    3. Compute scores for positive + negatives
    4. Rank and compute metrics
    
    Args:
        model: Recommendation model with forward() method
        ground_truth: {user_idx: set of positive test items}
        train_items: {user_idx: set of training items to exclude}
        num_items: Total number of items
        num_neg_samples: Number of negative samples (default 99 for 100 total)
        k_values: K values for @K metrics
        edge_index: Graph edge index for GNN models
        edge_weight: Edge weights
        device: Torch device
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    # Precompute embeddings for efficiency (only for graph models)
    user_emb = None
    item_emb = None
    
    # Check if model is a graph model (has forward with edge_index)
    model_type = type(model).__name__
    is_graph_model = model_type in ['LightGCN', 'NGCF']
    
    if is_graph_model and edge_index is not None:
        with torch.no_grad():
            edge_index = edge_index.to(device)
            if edge_weight is not None:
                edge_weight = edge_weight.to(device)
            user_emb, item_emb = model.forward(edge_index, edge_weight)
    
    test_users = sorted(ground_truth.keys())
    
    # Results accumulators
    hits_at_k = {k: [] for k in k_values}
    ndcg_at_k = {k: [] for k in k_values}
    precision_at_k = {k: [] for k in k_values}
    recall_at_k = {k: [] for k in k_values}
    
    print(f"Sampled evaluation: {len(test_users)} users, {num_neg_samples} neg samples each")
    
    for user_idx in tqdm(test_users, desc="Sampled Eval"):
        pos_items = list(ground_truth[user_idx])
        if not pos_items:
            continue
            
        # Get items to exclude from negative sampling
        exclude = train_items.get(user_idx, set()) | set(pos_items)
        
        # Sample negative items
        all_items = set(range(num_items))
        candidate_negatives = list(all_items - exclude)
        
        if len(candidate_negatives) < num_neg_samples:
            neg_items = candidate_negatives
        else:
            neg_items = np.random.choice(candidate_negatives, num_neg_samples, replace=False).tolist()
        
        # Create candidate list: positives + negatives
        candidates = pos_items + neg_items
        candidates_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
        
        # Compute scores based on model type
        if is_graph_model and user_emb is not None:
            # Graph model: use precomputed embeddings
            user_vec = user_emb[user_idx]  # (emb_dim,)
            item_vecs = item_emb[candidates_tensor]  # (num_candidates, emb_dim)
            scores = torch.matmul(item_vecs, user_vec)  # (num_candidates,)
        else:
            # Neural model (NCF, etc): use forward pass
            user_tensor = torch.tensor([user_idx] * len(candidates), dtype=torch.long, device=device)
            scores = model.forward(user_tensor, candidates_tensor).squeeze()  # (num_candidates,)
        
        # Get ranking (higher is better)
        _, indices = torch.sort(scores, descending=True)
        ranked_items = [candidates[i] for i in indices.cpu().tolist()]
        
        # Compute metrics
        for k in k_values:
            top_k = ranked_items[:k]
            
            # Count positives in top-K
            num_hits = sum(1 for item in top_k if item in pos_items)
            
            # Hit@K (any positive in top-K?)
            hit = num_hits > 0
            hits_at_k[k].append(1.0 if hit else 0.0)
            
            # Precision@K = hits / K
            precision_at_k[k].append(num_hits / k)
            
            # Recall@K = hits / num_positives
            recall_at_k[k].append(num_hits / len(pos_items))
            
            # NDCG@K
            dcg = 0.0
            for rank, item in enumerate(top_k):
                if item in pos_items:
                    dcg += 1.0 / np.log2(rank + 2)
            
            # IDCG (ideal: all positives at top)
            idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(pos_items), k)))
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_at_k[k].append(ndcg)
    
    # Aggregate results
    results = {}
    for k in k_values:
        results[f"hit_rate_at_{k}"] = np.mean(hits_at_k[k])
        results[f"ndcg_at_{k}"] = np.mean(ndcg_at_k[k])
        results[f"precision_at_{k}"] = np.mean(precision_at_k[k])
        results[f"recall_at_{k}"] = np.mean(recall_at_k[k])
    
    return results
