"""
Item-based Collaborative Filtering.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.base import BaseRecommender


class ItemBasedCF(BaseRecommender):
    """
    Item-based Collaborative Filtering using cosine similarity.
    
    Computes item-item similarity matrix and uses it to make predictions
    based on user's historical interactions.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        k_neighbors: int = 50,
        similarity: str = "cosine",
        device: str = "cpu",
    ):
        """
        Initialize Item-based CF.
        
        Args:
            num_users: Number of users
            num_items: Number of items
            k_neighbors: Number of neighbors to consider
            similarity: Similarity metric ("cosine", "pearson", "adjusted_cosine")
            device: Device for computations
        """
        super().__init__(num_users, num_items, device)
        
        self.k_neighbors = k_neighbors
        self.similarity = similarity
        
        # Will be computed during training
        self.item_similarity: np.ndarray | None = None
        self.user_item_matrix: np.ndarray | None = None
    
    def fit(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray | None = None,
    ) -> None:
        """
        Fit the model by computing item similarity matrix.
        
        Args:
            user_ids: Array of user indices
            item_ids: Array of item indices
            ratings: Optional array of ratings (1 if implicit)
        """
        from scipy.sparse import csr_matrix
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Build user-item matrix
        if ratings is None:
            ratings = np.ones(len(user_ids))
        
        self.user_item_matrix = csr_matrix(
            (ratings, (user_ids, item_ids)),
            shape=(self.num_users, self.num_items)
        )
        
        # Compute item-item similarity
        item_matrix = self.user_item_matrix.T.toarray()
        
        if self.similarity == "cosine":
            self.item_similarity = cosine_similarity(item_matrix)
        elif self.similarity == "pearson":
            # Mean-center each item
            item_means = np.mean(item_matrix, axis=1, keepdims=True)
            centered = item_matrix - item_means
            self.item_similarity = cosine_similarity(centered)
        elif self.similarity == "adjusted_cosine":
            # Mean-center by user
            user_means = self.user_item_matrix.mean(axis=1).A.flatten()
            adjusted = item_matrix - user_means[np.newaxis, :]
            self.item_similarity = cosine_similarity(adjusted)
        else:
            raise ValueError(f"Unknown similarity: {self.similarity}")
        
        # Zero out diagonal
        np.fill_diagonal(self.item_similarity, 0)
    
    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute prediction scores.
        
        Args:
            users: User indices
            items: Item indices (if None, score all items)
            
        Returns:
            Prediction scores
        """
        if self.item_similarity is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        users_np = users.cpu().numpy()
        
        # Get user's interaction vector
        user_vectors = self.user_item_matrix[users_np].toarray()
        
        if items is not None:
            items_np = items.cpu().numpy()
            # Point-wise predictions
            scores = []
            for i, (u, item) in enumerate(zip(users_np, items_np)):
                user_vec = user_vectors[i]
                # Get k most similar items that user has interacted with
                sim = self.item_similarity[item]
                interacted = np.where(user_vec > 0)[0]
                
                if len(interacted) == 0:
                    scores.append(0.0)
                    continue
                
                # Get top-k neighbors among interacted items
                neighbor_sims = sim[interacted]
                k = min(self.k_neighbors, len(interacted))
                top_k_idx = np.argsort(neighbor_sims)[-k:]
                
                # Weighted average
                weights = neighbor_sims[top_k_idx]
                if np.sum(weights) > 0:
                    score = np.sum(weights * user_vec[interacted[top_k_idx]]) / np.sum(np.abs(weights))
                else:
                    score = 0.0
                scores.append(score)
            
            return torch.tensor(scores, device=self.device)
        else:
            # Score all items for each user
            # Efficient matrix multiplication
            scores = np.dot(user_vectors, self.item_similarity.T)
            
            # Normalize by sum of similarities
            denom = np.sum(np.abs(self.item_similarity), axis=1)
            denom[denom == 0] = 1  # Avoid division by zero
            scores = scores / denom
            
            return torch.tensor(scores, device=self.device)
    
    def compute_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Item-CF doesn't use gradient-based training."""
        raise NotImplementedError("Item-CF uses fit() for training")
    
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
        exclude_interacted: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate top-K recommendations.
        
        Args:
            users: User indices
            top_k: Number of recommendations
            exclude_interacted: Whether to exclude already interacted items
            
        Returns:
            Top-K item indices for each user
        """
        scores = self.forward(users)
        
        if exclude_interacted:
            users_np = users.cpu().numpy()
            user_vectors = self.user_item_matrix[users_np].toarray()
            # Set interacted items to very low score
            scores = scores.cpu().numpy()
            scores[user_vectors > 0] = -np.inf
            scores = torch.tensor(scores, device=self.device)
        
        _, indices = torch.topk(scores, top_k, dim=-1)
        return indices
    
    def get_similar_items(
        self,
        item_id: int,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Get most similar items to a given item.
        
        Args:
            item_id: Item index
            top_k: Number of similar items
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if self.item_similarity is None:
            raise RuntimeError("Model not trained")
        
        similarities = self.item_similarity[item_id]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def save(self, path: Path) -> None:
        """Save model."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "item_similarity": self.item_similarity,
                "user_item_matrix": self.user_item_matrix,
                "config": {
                    "k_neighbors": self.k_neighbors,
                    "similarity": self.similarity,
                },
            }, f)
    
    def load(self, path: Path) -> None:
        """Load model."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.item_similarity = data["item_similarity"]
        self.user_item_matrix = data["user_item_matrix"]
    
    def __repr__(self) -> str:
        return f"ItemBasedCF(k_neighbors={self.k_neighbors}, similarity={self.similarity})"
