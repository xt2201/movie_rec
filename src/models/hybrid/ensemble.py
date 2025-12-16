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
                        user_emb, item_emb = model.forward(self.edge_index, self.edge_weight)
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
        Score-normalized weighted average fusion.
        
        Computes scores from each model, normalizes to [0, 1], then combines.
        Uses pre-computed embeddings for graph models.
        """
        num_users = len(users)
        device = users.device
        
        # Initialize
        combined_scores = torch.zeros((num_users, self.num_items), device=device)
        total_weight = 0.0
        models_used = 0
        
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
                            user_emb_batch = user_emb[users]
                            scores = torch.matmul(user_emb_batch, item_emb.T)
                        elif self.edge_index is not None:
                            # Compute on-the-fly if not pre-computed
                            user_emb, item_emb = model.forward(self.edge_index, self.edge_weight)
                            user_emb_batch = user_emb[users]
                            scores = torch.matmul(user_emb_batch, item_emb.T)
                        else:
                            print(f"Warning: {name} requires edge_index, skipping")
                            continue
                    
                    elif model_type == 'NCF':
                        # NCF - compute scores for all items per user
                        all_items = torch.arange(self.num_items, device=device)
                        batch_scores = []
                        for user in users:
                            user_exp = user.unsqueeze(0).expand(self.num_items)
                            score = torch.sigmoid(model.forward(user_exp, all_items).squeeze())
                            batch_scores.append(score)
                        scores = torch.stack(batch_scores)
                    
                    elif model_type == 'ItemBasedCF':
                        # ItemCF - uses its own forward
                        scores = model.forward(users)
                    
                    elif model_type == 'SVDRecommender':
                        # SVD - uses its own forward
                        scores = model.forward(users)
                    
                    else:
                        # Generic fallback
                        try:
                            scores = model.forward(users, items, **kwargs)
                        except Exception:
                            continue
                    
                    # Normalize scores to [0, 1]
                    if scores.numel() > 0:
                        scores_min = scores.min()
                        scores_max = scores.max()
                        if scores_max > scores_min:
                            scores = (scores - scores_min) / (scores_max - scores_min)
                        else:
                            scores = torch.zeros_like(scores)
                    
                    combined_scores += weight * scores
                    total_weight += weight
                    models_used += 1
                    
            except Exception as e:
                print(f"Warning: Could not compute scores from {name}: {e}")
                continue
        
        if models_used == 0:
            print("Warning: No models contributed to hybrid scores")
        
        # Normalize by total weight
        if total_weight > 0:
            combined_scores /= total_weight
        
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
        Generate top-K recommendations.
        
        Args:
            users: User indices
            k: Number of recommendations
            exclude_items: Dict mapping user_idx -> set of item_idx to exclude
            
        Returns:
            Tuple of (top-K scores, top-K item indices) for each user
        """
        scores = self.forward(users, **kwargs)
        
        # Exclude items if specified
        if exclude_items is not None:
            for i, user_id in enumerate(users.tolist()):
                if user_id in exclude_items:
                    for item_id in exclude_items[user_id]:
                        if item_id < self.num_items:
                            scores[i, item_id] = float("-inf")
        
        # Ensure k doesn't exceed num_items
        k = min(k, self.num_items)
        top_scores, indices = torch.topk(scores, k, dim=-1)
        return top_scores, indices
    
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
