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
        """
        super().__init__(num_users, num_items, device)
        
        self.models = models or {}
        self.weights = weights or {}
        self.fusion_method = fusion_method
        self.learn_weights = learn_weights
        
        # Learnable weights
        if learn_weights and models:
            self.weight_params = nn.Parameter(
                torch.ones(len(models)) / len(models)
            )
        else:
            self.weight_params = None
    
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
    
    def remove_model(self, name: str) -> None:
        """Remove a model from the ensemble."""
        if name in self.models:
            del self.models[name]
            del self.weights[name]
            
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
        """Weighted average of model scores."""
        combined_scores = None
        
        for name, model in self.models.items():
            weight = weights.get(name, 0)
            if weight == 0:
                continue
            
            with torch.no_grad():
                scores = model.forward(users, items, **kwargs)
            
            # Normalize scores to [0, 1]
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            
            if combined_scores is None:
                combined_scores = weight * scores
            else:
                combined_scores = combined_scores + weight * scores
        
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
        
        RRF_score(item) = sum over models of: weight / (k + rank(item))
        """
        # Get rankings from each model
        all_rankings = []
        model_weights = []
        
        for name, model in self.models.items():
            weight = weights.get(name, 0)
            if weight == 0:
                continue
            
            with torch.no_grad():
                scores = model.forward(users, items, **kwargs)
            
            # Convert scores to rankings
            rankings = torch.argsort(torch.argsort(scores, descending=True))
            all_rankings.append(rankings)
            model_weights.append(weight)
        
        # Compute RRF scores
        combined_scores = torch.zeros_like(all_rankings[0], dtype=torch.float)
        
        for rankings, weight in zip(all_rankings, model_weights):
            combined_scores = combined_scores + weight / (k + rankings.float())
        
        return combined_scores
    
    def _borda_count(
        self,
        users: torch.Tensor,
        items: torch.Tensor | None,
        weights: dict[str, float],
        **kwargs,
    ) -> torch.Tensor:
        """
        Borda count fusion.
        
        Each model assigns points based on ranking (n - rank).
        """
        all_rankings = []
        model_weights = []
        
        for name, model in self.models.items():
            weight = weights.get(name, 0)
            if weight == 0:
                continue
            
            with torch.no_grad():
                scores = model.forward(users, items, **kwargs)
            
            # Convert scores to rankings (lower rank = better)
            rankings = torch.argsort(torch.argsort(scores, descending=True))
            all_rankings.append(rankings)
            model_weights.append(weight)
        
        # Compute Borda scores
        n_items = all_rankings[0].shape[-1]
        combined_scores = torch.zeros_like(all_rankings[0], dtype=torch.float)
        
        for rankings, weight in zip(all_rankings, model_weights):
            # Borda points = n - rank
            borda_points = n_items - rankings.float()
            combined_scores = combined_scores + weight * borda_points
        
        return combined_scores
    
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
        top_k: int = 10,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate top-K recommendations.
        
        Args:
            users: User indices
            top_k: Number of recommendations
            
        Returns:
            Top-K item indices for each user
        """
        scores = self.forward(users, **kwargs)
        _, indices = torch.topk(scores, top_k, dim=-1)
        return indices
    
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
