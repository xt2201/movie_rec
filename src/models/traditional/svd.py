"""
SVD-based recommendation using Surprise library.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.base import BaseRecommender


class SVDRecommender(BaseRecommender):
    """
    SVD-based collaborative filtering using Surprise library.
    
    Wraps the Surprise SVD implementation in our BaseRecommender interface.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        device: str = "cpu",
    ):
        """
        Initialize SVD recommender.
        
        Args:
            num_users: Number of users
            num_items: Number of items
            n_factors: Number of latent factors
            n_epochs: Number of training epochs
            lr_all: Learning rate for all parameters
            reg_all: Regularization for all parameters
            device: Device (not used, Surprise is CPU-only)
        """
        super().__init__(num_users, num_items, device)
        
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        
        self.model = None
        self.trainset = None
        
        # Store embeddings after training
        self._user_embeddings: np.ndarray | None = None
        self._item_embeddings: np.ndarray | None = None
    
    def _build_model(self):
        """Build Surprise SVD model."""
        try:
            from surprise import SVD
        except ImportError:
            raise ImportError("Please install scikit-surprise: pip install scikit-surprise")
        
        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
        )
    
    def fit(self, trainset):
        """
        Fit the SVD model.
        
        Args:
            trainset: Surprise trainset object
        """
        if self.model is None:
            self._build_model()
        
        self.trainset = trainset
        self.model.fit(trainset)
        
        # Extract embeddings
        self._user_embeddings = self.model.pu
        self._item_embeddings = self.model.qi
    
    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass - compute predictions.
        
        Args:
            users: User indices
            items: Item indices (if None, score all items)
            
        Returns:
            Prediction scores
        """
        if self._user_embeddings is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        users_np = users.cpu().numpy()
        
        if items is not None:
            items_np = items.cpu().numpy()
            # Point-wise predictions
            scores = np.sum(
                self._user_embeddings[users_np] * self._item_embeddings[items_np],
                axis=1
            )
        else:
            # Score all items for each user
            scores = np.dot(
                self._user_embeddings[users_np],
                self._item_embeddings.T
            )
        
        return torch.tensor(scores, device=self.device)
    
    def compute_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """SVD doesn't use this loss - training is done via fit()."""
        raise NotImplementedError("SVD uses internal training via fit()")
    
    def predict(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Predict scores for user-item pairs."""
        return self.forward(users, items)
    
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
        scores = self.forward(users)
        _, indices = torch.topk(scores, top_k, dim=-1)
        return indices
    
    def get_embeddings(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Get user and item embeddings."""
        if self._user_embeddings is None:
            raise RuntimeError("Model not trained")
        
        return (
            torch.tensor(self._user_embeddings, device=self.device),
            torch.tensor(self._item_embeddings, device=self.device),
        )
    
    def save(self, path: Path) -> None:
        """Save model."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "user_embeddings": self._user_embeddings,
                "item_embeddings": self._item_embeddings,
                "config": {
                    "n_factors": self.n_factors,
                    "n_epochs": self.n_epochs,
                    "lr_all": self.lr_all,
                    "reg_all": self.reg_all,
                },
            }, f)
    
    def load(self, path: Path) -> None:
        """Load model."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.model = data["model"]
        self._user_embeddings = data["user_embeddings"]
        self._item_embeddings = data["item_embeddings"]
    
    def __repr__(self) -> str:
        return f"SVDRecommender(n_factors={self.n_factors})"
