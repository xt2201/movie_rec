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
        
        For implicit feedback: score(u, i) = sum of sim(i, j) for j in user's items
        
        Args:
            users: User indices
            items: Item indices (if None, score all items)
            
        Returns:
            Prediction scores
        """
        if self.item_similarity is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        users_np = users.cpu().numpy()
        
        # Get user's interaction vector (binary: 1 if interacted, 0 otherwise)
        user_vectors = self.user_item_matrix[users_np].toarray()
        # Binarize to ensure implicit feedback
        user_binary = (user_vectors > 0).astype(np.float32)
        
        if items is not None:
            items_np = items.cpu().numpy()
            # Point-wise predictions using top-k neighbors
            scores = []
            for i, (u, item) in enumerate(zip(users_np, items_np)):
                user_vec = user_binary[i]
                sim = self.item_similarity[item]
                interacted = np.where(user_vec > 0)[0]
                
                if len(interacted) == 0:
                    scores.append(0.0)
                    continue
                
                # Get top-k similar items among user's interacted items
                neighbor_sims = sim[interacted]
                k = min(self.k_neighbors, len(interacted))
                top_k_idx = np.argpartition(neighbor_sims, -k)[-k:]
                
                # Score = sum of top-k similarities
                score = np.sum(neighbor_sims[top_k_idx])
                scores.append(score)
            
            return torch.tensor(scores, device=self.device, dtype=torch.float32)
        else:
            # Score all items for each user
            # For each (user, item): score = sum of sim(item, j) for j in top-k similar items user has
            batch_size = len(users_np)
            all_scores = np.zeros((batch_size, self.num_items), dtype=np.float32)
            
            for i in range(batch_size):
                interacted = np.where(user_binary[i] > 0)[0]
                
                if len(interacted) == 0:
                    continue
                
                k = min(self.k_neighbors, len(interacted))
                
                # For each item, get similarity to user's interacted items
                # sim_subset shape: (num_items, num_interacted)
                sim_subset = self.item_similarity[:, interacted]
                
                # For each candidate item, sum top-k similarities
                if k >= len(interacted):
                    # Use all interacted items
                    all_scores[i] = np.sum(sim_subset, axis=1)
                else:
                    # Use only top-k most similar for each item
                    # Partition to get top-k indices, then sum those
                    top_k_indices = np.argpartition(sim_subset, -k, axis=1)[:, -k:]
                    # Gather top-k values
                    row_idx = np.arange(self.num_items)[:, np.newaxis]
                    top_k_sims = sim_subset[row_idx, top_k_indices]
                    all_scores[i] = np.sum(top_k_sims, axis=1)
            
            return torch.tensor(all_scores, device=self.device, dtype=torch.float32)
    
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate top-K recommendations.
        
        Args:
            users: User indices
            top_k: Number of recommendations
            exclude_interacted: Whether to exclude already interacted items
            
        Returns:
            Tuple of (top-K scores, top-K item indices) for each user
        """
        scores = self.forward(users)
        
        if exclude_interacted:
            users_np = users.cpu().numpy()
            user_vectors = self.user_item_matrix[users_np].toarray()
            # Set interacted items to very low score
            scores = scores.cpu().numpy()
            scores[user_vectors > 0] = -np.inf
            scores = torch.tensor(scores, device=self.device)
        
        top_scores, indices = torch.topk(scores, top_k, dim=-1)
        return top_scores, indices
    
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
