"""
Hybrid Ensemble recommendation model.

Combines multiple recommendation models using weighted ensemble.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.models.base import BaseRecommender


class HybridEnsemble(BaseRecommender):
    """
    Hybrid ensemble combining multiple recommendation models.
    
    Supports:
    - Weighted average of scores
    - Rank fusion (RRF, Borda count)
    - Learned weights via neural network
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        models: dict[str, BaseRecommender] | None = None,
        weights: dict[str, float] | None = None,
        fusion_method: str = "weighted_average",
        learn_weights: bool = False,
        k_rrf: int = 60,
        device: str = "cpu",
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ):
        """
        Initialize hybrid ensemble.
        
        Args:
            num_users: Number of users
            num_items: Number of items
            models: Dictionary of model name -> model instance
            weights: Dictionary of model name -> weight (must sum to 1)
            fusion_method: How to combine scores ("weighted_average", "rrf", "borda")
            learn_weights: Whether to learn weights via backprop
            device: Device for computations
            edge_index: Graph edge index for graph-based models
            edge_weight: Graph edge weights
        """
        super().__init__(num_users, num_items, device)
        
        self.models = models or {}
        self.weights = weights or {}
        self.fusion_method = fusion_method
        self.learn_weights = learn_weights
        self.k_rrf = k_rrf  # RRF constant (lower = more emphasis on top ranks)
        
        # Store edge_index for graph models
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        
        # Pre-computed embeddings for graph models (computed once)
        self._graph_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        
        # Pre-compute embeddings for graph models if edge_index provided
        if edge_index is not None and models:
            self._precompute_graph_embeddings()
        
        # Learnable weights
        if learn_weights and models:
            self.weight_params = nn.Parameter(
                torch.ones(len(models)) / len(models)
            )
        else:
            self.weight_params = None
    
    def _precompute_graph_embeddings(self) -> None:
        """Pre-compute embeddings from graph models to avoid repeated forward passes."""
        if self.edge_index is None:
            return
            
        for name, model in self.models.items():
            model_type = type(model).__name__
            if model_type in ['LightGCN', 'NGCF']:
                try:
                    with torch.no_grad():
                        # Move edge_index to model device
                        model_device = next(model.parameters()).device
                        edge_index_dev = self.edge_index.to(model_device)
                        edge_weight_dev = self.edge_weight.to(model_device) if self.edge_weight is not None else None
                        
                        user_emb, item_emb = model.forward(edge_index_dev, edge_weight_dev)
                        self._graph_embeddings[name] = (user_emb, item_emb)
                except Exception as e:
                    print(f"Warning: Could not pre-compute embeddings for {name}: {e}")
    
    def add_model(
        self,
        name: str,
        model: BaseRecommender,
        weight: float = 1.0,
    ) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            name: Model name
            model: Model instance
            weight: Model weight
        """
        self.models[name] = model
        self.weights[name] = weight
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        
        # Pre-compute embeddings if graph model
        model_type = type(model).__name__
        if model_type in ['LightGCN', 'NGCF'] and self.edge_index is not None:
            try:
                with torch.no_grad():
                    user_emb, item_emb = model.forward(self.edge_index, self.edge_weight)
                    self._graph_embeddings[name] = (user_emb, item_emb)
            except Exception as e:
                print(f"Warning: Could not compute embeddings for {name}: {e}")
    
    def remove_model(self, name: str) -> None:
        """Remove a model from the ensemble."""
        if name in self.models:
            del self.models[name]
            del self.weights[name]
            if name in self._graph_embeddings:
                del self._graph_embeddings[name]
            
            # Renormalize
            if self.weights:
                total = sum(self.weights.values())
                self.weights = {k: v / total for k, v in self.weights.items()}
    
    def set_weights(self, weights: dict[str, float]) -> None:
        """
        Set model weights manually.
        
        Args:
            weights: Dictionary of model_name -> weight
        """
        self.weights = weights.copy()
        # Normalize
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def optimize_weights(
        self,
        val_users: list[int],
        ground_truth: dict[int, set[int]],
        train_items: dict[int, set[int]],
        k: int = 10,
        weight_steps: int = 11,
    ) -> dict[str, float]:
        """
        Optimize ensemble weights using grid search on validation set.
        
        Args:
            val_users: List of validation user IDs
            ground_truth: Dict of user -> set of relevant items
            train_items: Dict of user -> set of training items to exclude
            k: Top-K for evaluation
            weight_steps: Number of weight values to try (0 to 1)
            
        Returns:
            Best weights found
        """
        import numpy as np
        from itertools import product
        
        import numpy as np
        
        model_names = list(self.models.keys())
        num_models = len(model_names)
        
        best_ndcg = -1
        best_weights = self.weights.copy()
        
        print(f"Optimizing weights for {model_names}...")
        
        # Convert users to tensor
        user_tensor = torch.tensor(val_users, dtype=torch.long, device=self.device)
        
        # Pre-compute recommendations for all models on validation set to speed up optimization
        # dict[model_name] -> (top_scores, top_items)
        cached_preds = {}
        valid_models = []
        
        print("  Pre-computing predictions for validation...")
        for name, model in self.models.items():
            try:
                with torch.no_grad():
                    model_type = type(model).__name__
                    
                    if model_type in ['LightGCN', 'NGCF']:
                        # Use pre-computed embeddings
                        if name not in self._graph_embeddings:
                            print(f"  Warning: No pre-computed embeddings for {name}, skipping")
                            continue
                        
                        user_emb, item_emb = self._graph_embeddings[name]
                        # Compute scores using embeddings
                        users_m = user_tensor.clamp(0, user_emb.size(0) - 1)
                        user_emb_batch = user_emb[users_m]
                        scores = torch.matmul(user_emb_batch, item_emb.T)
                        top_scores, items = torch.topk(scores, k=k, dim=-1)
                        cached_preds[name] = (top_scores.cpu(), items.cpu())
                    
                    elif type(model).__name__ == 'NCF':
                         d = next(model.parameters()).device
                         users_m = user_tensor.to(d)
                         scores, items = model.recommend(users_m, k=k, exclude_items=train_items)
                         cached_preds[name] = (scores.cpu(), items.cpu())
                    elif type(model).__name__ == 'SVDRecommender':
                         # SVD input must be CPU
                         users_m = user_tensor.cpu()
                         scores, items = model.recommend(users_m, k=k, exclude_items=train_items)
                         cached_preds[name] = (scores.cpu(), items.cpu())
                    elif type(model).__name__ == 'ItemBasedCF':
                         users_m = user_tensor.cpu()
                         scores, items = model.recommend(users_m, k=k, exclude_items=train_items)
                         cached_preds[name] = (scores.cpu(), items.cpu())
                    else:
                        # Generic fallback - try to infer device
                        try:
                            d = next(model.parameters()).device
                            users_m = user_tensor.to(d)
                        except StopIteration:
                            users_m = user_tensor
                        scores, items = model.recommend(users_m, k=k, exclude_items=train_items)
                        cached_preds[name] = (scores.cpu(), items.cpu())
                    
                    valid_models.append(name)
            except Exception as e:
                print(f"  Warning: Failed to get preds for {name}: {e}")
        
        if not valid_models:
            return self.weights

        # Random search settings - more trials for better exploration
        num_trials = 100 if num_models <= 2 else 200
        
        # Always try equal weights first
        candidates = [np.ones(num_models) / num_models]
        
        # Add single-model weights (1.0 for one, 0 for others)
        eye = np.eye(num_models)
        candidates.extend([eye[i] for i in range(num_models)])
        
        # Add biased candidates (high weight on one model, rest split)
        for i in range(num_models):
            for alpha in [0.6, 0.7, 0.8, 0.9, 0.95]:
                w = np.ones(num_models) * (1.0 - alpha) / max(1, num_models - 1)
                w[i] = alpha
                candidates.append(w)
        
        # Add pairwise combinations (50% each for top pairs)
        for i in range(num_models):
            for j in range(i + 1, num_models):
                w = np.zeros(num_models)
                w[i] = 0.5
                w[j] = 0.5
                candidates.append(w)
        
        # Add random candidates with various concentrations
        for _ in range(num_trials):
            # Dirichlet with alpha=0.5 biases toward sparse (dominant) weights
            w = np.random.dirichlet(np.ones(num_models) * 0.5)
            candidates.append(w)
            
        k_rrf = self.k_rrf
        
        for idx, w_vec in enumerate(candidates):
            current_weights = {name: val for name, val in zip(valid_models, w_vec)}
            
            # Fast Ensemble scoring using cached predictions (Approximate RRF)
            # Full RRF requires sorting potentially all items. 
            # Here we only have Top-K from each model. 
            # We will merge these Top-K lists.
            
            # This is an approximation: we only consider items that appeared in at least one model's top K
            # For validation optimization, this is usually sufficient proxy
            
            ndcg_sum = 0.0
            
            # Vectorized scoring would be hard due to variable item sets
            # Loop over users
            for u_idx, user_id in enumerate(val_users):
                if user_id not in ground_truth:
                    continue
                
                # Combine scores for this user
                item_scores = {}
                
                for name_idx, name in enumerate(valid_models):
                    weight = w_vec[name_idx]
                    if weight <= 1e-6: continue
                    
                    # preds is (scores_tensor, items_tensor)
                    _, m_items = cached_preds[name]
                    # m_items is (num_users, k)
                    
                    user_items = m_items[u_idx].tolist()
                    
                    for rank, item_idx in enumerate(user_items):
                        rrf_score = weight / (k_rrf + rank)
                        item_scores[item_idx] = item_scores.get(item_idx, 0.0) + rrf_score
                
                # Sort combined
                sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:k]
                predicted_items = [x[0] for x in sorted_items]
                
                # Calc NDCG
                relevant = ground_truth[user_id]
                dcg = 0.0
                for rank, item in enumerate(predicted_items):
                    if item in relevant:
                        dcg += 1.0 / np.log2(rank + 2)
                
                idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), k)))
                ndcg_sum += dcg / idcg if idcg > 0 else 0.0
            
            avg_ndcg = ndcg_sum / len(val_users)
            
            if avg_ndcg > best_ndcg:
                best_ndcg = avg_ndcg
                best_weights = current_weights.copy()
                if idx % 10 == 0:
                   print(f"  Trial {idx}: Best NDCG@{k}={avg_ndcg:.4f}")

        print(f"Best weights: {best_weights}")
        return best_weights
    
    def _get_weights(self) -> dict[str, float]:
        """Get current weights (learned or fixed)."""
        if self.learn_weights and self.weight_params is not None:
            # Softmax over learned weights
            softmax_weights = torch.softmax(self.weight_params, dim=0)
            return {
                name: softmax_weights[i].item()
                for i, name in enumerate(self.models.keys())
            }
        return self.weights
    
    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute ensemble predictions.
        
        Args:
            users: User indices
            items: Item indices (if None, score all items)
            
        Returns:
            Ensemble prediction scores
        """
        if not self.models:
            raise RuntimeError("No models in ensemble")
        
        weights = self._get_weights()
        
        if self.fusion_method == "weighted_average":
            return self._weighted_average(users, items, weights, **kwargs)
        elif self.fusion_method == "rrf":
            return self._reciprocal_rank_fusion(users, items, weights, **kwargs)
        elif self.fusion_method == "borda":
            return self._borda_count(users, items, weights, **kwargs)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _weighted_average(
        self,
        users: torch.Tensor,
        items: torch.Tensor | None,
        weights: dict[str, float],
        **kwargs,
    ) -> torch.Tensor:
        """
        RRF-based weighted fusion using model.recommend().
        
        Instead of computing all-item scores (memory-intensive),
        uses recommend() from each model and combines via RRF.
        """
        num_users = len(users)
        device = users.device
        k_rrf = 60  # RRF constant
        max_k = min(100, self.num_items)  # Top-K from each model
        
        # Initialize combined scores
        combined_scores = torch.zeros((num_users, self.num_items), device=device)
        models_used = 0
        
        # Flatten users if needed
        users_flat = users.flatten() if users.dim() > 1 else users
        
        for name, model in self.models.items():
            weight = weights.get(name, 0)
            if weight == 0:
                continue
            
            try:
                with torch.no_grad():
                    model_type = type(model).__name__
                    
                    if model_type in ['LightGCN', 'NGCF']:
                        # Use pre-computed embeddings for graph models
                        if name in self._graph_embeddings:
                            user_emb, item_emb = self._graph_embeddings[name]
                            user_emb_batch = user_emb[users_flat]
                            scores = torch.matmul(user_emb_batch, item_emb.T)
                            # Get top-K
                            top_scores, top_items = torch.topk(scores, max_k, dim=-1)
                        elif self.edge_index is not None:
                            user_emb, item_emb = model.forward(self.edge_index, self.edge_weight)
                            user_emb_batch = user_emb[users_flat]
                            scores = torch.matmul(user_emb_batch, item_emb.T)
                            top_scores, top_items = torch.topk(scores, max_k, dim=-1)
                        else:
                            print(f"Warning: {name} requires edge_index, skipping")
                            continue
                    
                    elif model_type == 'NCF':
                        # Use NCF's recommend method - clamp user indices
                        max_user = model.num_users - 1
                        users_clamped = torch.clamp(users_flat, 0, max_user)
                        users_ncf = users_clamped.to(next(model.parameters()).device)
                        top_scores, top_items = model.recommend(users_ncf, k=max_k)
                        top_items = top_items.to(device)
                    
                    elif model_type == 'ItemBasedCF':
                        # Use ItemCF's recommend method - clamp user indices
                        users_cpu = users_flat.cpu()
                        max_user = model.num_users - 1
                        users_clipped = torch.clamp(users_cpu, 0, max_user)
                        top_scores, top_items = model.recommend(users_clipped, k=max_k)
                        top_items = top_items.to(device)
                    
                    elif model_type == 'SVDRecommender':
                        # Use SVD's recommend method - clamp user indices
                        max_user = model.num_users - 1
                        users_clamped = torch.clamp(users_flat, 0, max_user)
                        model_device = next(model.parameters()).device
                        users_svd = users_clamped.to(model_device)
                        top_scores, top_items = model.recommend(users_svd, k=max_k)
                        top_items = top_items.to(device)
                    
                    else:
                        # Generic fallback
                        try:
                            top_scores, top_items = model.recommend(users_flat, k=max_k)
                            top_items = top_items.to(device)
                        except Exception:
                            continue
                    
                    # Convert to RRF scores
                    for i in range(num_users):
                        for rank, item_idx in enumerate(top_items[i].tolist()):
                            if 0 <= item_idx < self.num_items:
                                rrf_score = weight / (k_rrf + rank)
                                combined_scores[i, item_idx] += rrf_score
                    
                    models_used += 1
                    
            except Exception as e:
                print(f"Warning: Could not get recommendations from {name}: {e}")
                continue
        
        if models_used == 0:
            print("Warning: No models contributed to hybrid scores")
        
        return combined_scores
    
    def _reciprocal_rank_fusion(
        self,
        users: torch.Tensor,
        items: torch.Tensor | None,
        weights: dict[str, float],
        k: int = 60,
        **kwargs,
    ) -> torch.Tensor:
        """
        Reciprocal Rank Fusion (RRF).
        
        Uses the same score computation as weighted average, then converts to RRF.
        """
        # Get scores using weighted average method
        scores = self._weighted_average(users, items, weights, **kwargs)
        
        # Convert to RRF scores
        rankings = torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1)
        rrf_scores = 1.0 / (k + rankings.float())
        
        return rrf_scores
    
    def _borda_count(
        self,
        users: torch.Tensor,
        items: torch.Tensor | None,
        weights: dict[str, float],
        **kwargs,
    ) -> torch.Tensor:
        """
        Borda count fusion.
        
        Uses the same score computation as weighted average, then converts to Borda.
        """
        # Get scores using weighted average method
        scores = self._weighted_average(users, items, weights, **kwargs)
        
        # Convert to Borda scores
        rankings = torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1)
        borda_scores = (self.num_items - rankings.float())
        
        return borda_scores
    
    def compute_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute loss for learning ensemble weights.
        
        Uses BPR loss on ensemble predictions.
        """
        if not self.learn_weights:
            raise RuntimeError("Ensemble weights are not learnable")
        
        pos_scores = self.forward(users, pos_items, **kwargs)
        neg_scores = self.forward(users, neg_items, **kwargs)
        
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        return loss
    
    def predict(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Predict scores for user-item pairs."""
        return self.forward(users, items, **kwargs)
    
    def recommend(
        self,
        users: torch.Tensor,
        k: int = 10,
        exclude_items: dict[int, set[int]] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate top-K recommendations using RRF fusion of base models.
        
        This method directly calls base models' recommend() and merges
        results via RRF, avoiding full score matrix allocation.
        Explicitly manages memory and devices to prevent OOM.
        
        Args:
            users: User indices
            k: Number of recommendations
            exclude_items: Dict mapping user_idx -> set of item_idx to exclude
            
        Returns:
            Tuple of (top-K scores, top-K item indices) for each user
        """
        # Force GC at start
        import gc
        import sys
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        num_users = len(users)
        device = users.device
        k_rrf = self.k_rrf  # Use configurable RRF constant
        max_k = min(100, self.num_items)  # Top-K from each model
        
        # Flatten users if needed
        users_flat = users.flatten() if users.dim() > 1 else users
        
        weights = self._get_weights()
        
        # Collect RRF scores for each item
        # Use dict to avoid full matrix: {user_idx: {item_idx: score}}
        user_item_scores = [{} for _ in range(num_users)]
        models_used = 0
        
        for name, model in self.models.items():
            weight = weights.get(name, 0)
            if weight == 0:
                continue
            
            try:
                # Clear previous iteration memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                with torch.no_grad():
                    model_type = type(model).__name__
                    
                    # Clamp user indices to valid range
                    max_user = model.num_users - 1
                    users_clamped = torch.clamp(users_flat, 0, max_user)
                    
                    if model_type in ['LightGCN', 'NGCF']:
                        if name in self._graph_embeddings:
                            user_emb, item_emb = self._graph_embeddings[name]
                            user_emb_batch = user_emb[users_clamped]
                            scores = torch.matmul(user_emb_batch, item_emb.T)
                            top_scores, top_items = torch.topk(scores, max_k, dim=-1)
                            del scores
                        elif self.edge_index is not None:
                            user_emb, item_emb = model.forward(self.edge_index, self.edge_weight)
                            user_emb_batch = user_emb[users_clamped]
                            scores = torch.matmul(user_emb_batch, item_emb.T)
                            top_scores, top_items = torch.topk(scores, max_k, dim=-1)
                            del scores, user_emb, item_emb, user_emb_batch
                        else:
                            # Fallback: call model.recommend directly with proper device
                            model_device = next(model.parameters()).device
                            users_m = users_clamped.to(model_device)
                            top_scores, top_items = model.recommend(users_m, k=max_k, exclude_items=exclude_items)
                            del users_m
                    
                    elif model_type == 'NCF':
                        # NCF: use proper device from model parameters
                        model_device = next(model.parameters()).device
                        users_ncf = users_clamped.to(model_device)
                        top_scores, top_items = model.recommend(users_ncf, k=max_k, exclude_items=exclude_items)
                        del users_ncf
                    
                    elif model_type == 'SVDRecommender':
                        # Force SVD to use CPU inputs since weights are on CPU
                        users_svd = users_clamped.cpu()
                        top_scores, top_items = model.recommend(users_svd, k=max_k, exclude_items=exclude_items)
                        del users_svd
                    
                    elif model_type == 'ItemBasedCF':
                        users_cpu = users_clamped.cpu()
                        # Ensure ItemCF runs on CPU
                        original_device = model._device
                        model._device = torch.device('cpu') 
                        try:
                            # ItemCF.recommend returns indices on device, so move result to CPU immediately
                            top_scores, top_items = model.recommend(users_cpu, k=max_k, exclude_items=exclude_items)
                            top_scores = top_scores.cpu()
                            top_items = top_items.cpu()
                        finally:
                            model._device = original_device
                        del users_cpu
                    
                    else:
                        # Generic fallback
                        try:
                            top_scores, top_items = model.recommend(users_clamped, k=max_k, exclude_items=exclude_items)
                        except Exception:
                            continue
                    
                    # Convert to RRF scores and accumulate
                    top_items_cpu = top_items.cpu()
                    del top_items, top_scores
                    
                    for i in range(num_users):
                        items_list = top_items_cpu[i].tolist()
                        for rank, item_idx in enumerate(items_list):
                            if 0 <= item_idx < self.num_items:
                                rrf_score = weight / (k_rrf + rank)
                                if item_idx not in user_item_scores[i]:
                                    user_item_scores[i][item_idx] = 0.0
                                user_item_scores[i][item_idx] += rrf_score
                    
                    del top_items_cpu
                    models_used += 1
                    
            except Exception as e:
                print(f"Warning: Could not get recommendations from {name}: {e}")
                continue
        
        if models_used == 0:
            print("Warning: No models contributed to hybrid recommendations")
            return (
                torch.zeros((num_users, k), device=device),
                torch.zeros((num_users, k), dtype=torch.long, device=device),
            )
        
        # Sort and get top-K for each user
        top_scores_list = []
        top_items_list = []
        
        for i in range(num_users):
            # Sort items by score descending
            sorted_items = sorted(
                user_item_scores[i].items(),
                key=lambda x: x[1],
                reverse=True
            )[:k]
            
            if len(sorted_items) < k:
                # Pad with zeros
                sorted_items.extend([(0, 0.0)] * (k - len(sorted_items)))
            
            items = [item for item, score in sorted_items]
            scores = [score for item, score in sorted_items]
            
            top_items_list.append(items)
            top_scores_list.append(scores)
            
        del user_item_scores
        gc.collect()
        
        return (
            torch.tensor(top_scores_list, dtype=torch.float32, device=device),
            torch.tensor(top_items_list, dtype=torch.long, device=device),
        )
    
    def save(self, path: Path) -> None:
        """Save ensemble configuration."""
        config = {
            "weights": self.weights,
            "fusion_method": self.fusion_method,
            "learn_weights": self.learn_weights,
            "model_names": list(self.models.keys()),
        }
        
        if self.learn_weights and self.weight_params is not None:
            config["weight_params"] = self.weight_params.data
        
        torch.save(config, path)
    
    def load(self, path: Path) -> None:
        """Load ensemble configuration."""
        config = torch.load(path, map_location=self.device)
        
        self.weights = config["weights"]
        self.fusion_method = config["fusion_method"]
        self.learn_weights = config["learn_weights"]
        
        if self.learn_weights and "weight_params" in config:
            self.weight_params = nn.Parameter(config["weight_params"])
    
    def __repr__(self) -> str:
        model_info = ", ".join(
            f"{name}({self.weights.get(name, 0):.2f})"
            for name in self.models.keys()
        )
        return f"HybridEnsemble(models=[{model_info}], fusion={self.fusion_method})"
