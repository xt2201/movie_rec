"""
BPR-MF (Matrix Factorization with BPR loss) using PyTorch GPU.

Optimized for maximum GPU utilization with pre-generated negative samples.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.models.base import BaseRecommender


class SVDRecommender(BaseRecommender):
    """
    BPR-optimized Matrix Factorization using PyTorch.
    
    Uses Bayesian Personalized Ranking loss for ranking optimization,
    with CUDA GPU acceleration and optimized batching.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        n_factors: int = 64,
        n_epochs: int = 100,
        lr: float = 0.05,
        reg: float = 0.0001,
        batch_size: int = 65536,
        device: str = "cuda",
    ):
        """
        Initialize BPR-MF recommender.
        
        Args:
            num_users: Number of users
            num_items: Number of items
            n_factors: Number of latent factors
            n_epochs: Number of training epochs
            lr: Learning rate
            reg: L2 regularization factor
            batch_size: Training batch size (large for GPU)
            device: Device (cuda/cpu)
        """
        super().__init__(num_users, num_items, device)
        
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        
        # PyTorch embeddings
        self.user_embedding = nn.Embedding(num_users, n_factors)
        self.item_embedding = nn.Embedding(num_items, n_factors)
        
        # Initialize weights
        self._init_weights()
        
        # Move to device
        self.to(self.device)
        
        # Training data
        self._user_items: dict[int, set[int]] | None = None
    
    def _init_weights(self) -> None:
        """Initialize embeddings with small random values."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def fit(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Fit the BPR-MF model using mini-batch SGD on GPU.
        
        Uses pre-generated negative samples for maximum GPU utilization.
        
        Args:
            user_ids: Array of user indices
            item_ids: Array of item indices
            ratings: Not used (implicit feedback)
            verbose: Whether to show progress bar
        """
        # Build user->items mapping for negative sampling
        self._user_items = {}
        for u, i in zip(user_ids, item_ids):
            if u not in self._user_items:
                self._user_items[u] = set()
            self._user_items[u].add(i)
        
        n_samples = len(user_ids)
        print(f"Training BPR-MF with {n_samples:,} interactions on {self.device}")
        print(f"Users: {self.num_users:,}, Items: {self.num_items:,}, Factors: {self.n_factors}")
        print(f"Batch size: {self.batch_size:,}, Epochs: {self.n_epochs}")
        
        # Convert to tensors and move to GPU
        users_tensor = torch.LongTensor(user_ids).to(self.device)
        pos_items_tensor = torch.LongTensor(item_ids).to(self.device)
        
        # Optimizer with weight decay for L2 regularization
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.reg
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.n_epochs,
            eta_min=self.lr * 0.01
        )
        
        # Training
        self.train()
        best_loss = float('inf')
        
        for epoch in range(self.n_epochs):
            # Pre-generate ALL negative samples for this epoch on GPU
            neg_items_tensor = self._generate_negatives_fast(users_tensor)
            
            # Shuffle data
            perm = torch.randperm(n_samples, device=self.device)
            users_shuffled = users_tensor[perm]
            pos_items_shuffled = pos_items_tensor[perm]
            neg_items_shuffled = neg_items_tensor[perm]
            
            total_loss = 0.0
            n_batches = (n_samples + self.batch_size - 1) // self.batch_size
            
            # Progress bar for epoch
            pbar = tqdm(
                range(0, n_samples, self.batch_size),
                desc=f"Epoch {epoch+1:3d}/{self.n_epochs}",
                disable=not verbose,
                leave=False,
                ncols=100,
            )
            
            for batch_start in pbar:
                batch_end = min(batch_start + self.batch_size, n_samples)
                
                # Get batch (already on GPU)
                batch_users = users_shuffled[batch_start:batch_end]
                batch_pos = pos_items_shuffled[batch_start:batch_end]
                batch_neg = neg_items_shuffled[batch_start:batch_end]
                
                # Compute BPR loss
                optimizer.zero_grad()
                
                # Get embeddings
                user_emb = self.user_embedding(batch_users)
                pos_item_emb = self.item_embedding(batch_pos)
                neg_item_emb = self.item_embedding(batch_neg)
                
                # Scores
                pos_scores = (user_emb * pos_item_emb).sum(dim=1)
                neg_scores = (user_emb * neg_item_emb).sum(dim=1)
                
                # BPR loss
                loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.5f}"
                })
            
            scheduler.step()
            avg_loss = total_loss / n_batches
            
            # Log every epoch
            if verbose:
                lr = scheduler.get_last_lr()[0]
                marker = " *" if avg_loss < best_loss else ""
                print(f"Epoch {epoch+1:3d}/{self.n_epochs} | Loss: {avg_loss:.4f} | LR: {lr:.5f}{marker}")
                best_loss = min(best_loss, avg_loss)
    
    def _generate_negatives_fast(self, users: torch.Tensor) -> torch.Tensor:
        """
        Generate negative samples for all users in batch.
        
        Uses vectorized random sampling with rejection for items in user history.
        """
        n = len(users)
        users_cpu = users.cpu().numpy()
        
        # Random sample negative items
        neg_items = np.random.randint(0, self.num_items, size=n)
        
        # Rejection sampling for items in user history
        for i in range(n):
            u = users_cpu[i]
            user_items = self._user_items.get(u, set())
            while neg_items[i] in user_items:
                neg_items[i] = np.random.randint(0, self.num_items)
        
        return torch.LongTensor(neg_items).to(self.device)
    
    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass - compute scores.
        
        Args:
            users: User indices
            items: Item indices (if None, score all items)
            
        Returns:
            Prediction scores
        """
        self.eval()
        with torch.no_grad():
            users = users.to(self.device)
            user_emb = self.user_embedding(users)
            
            if items is not None:
                items = items.to(self.device)
                item_emb = self.item_embedding(items)
                scores = (user_emb * item_emb).sum(dim=-1)
            else:
                # Score all items
                all_item_emb = self.item_embedding.weight
                scores = torch.matmul(user_emb, all_item_emb.T)
            
            return scores
    
    def compute_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute BPR loss."""
        users = users.to(self.device)
        pos_items = pos_items.to(self.device)
        neg_items = neg_items.to(self.device)
        
        user_emb = self.user_embedding(users)
        pos_item_emb = self.item_embedding(pos_items)
        neg_item_emb = self.item_embedding(neg_items)
        
        pos_scores = (user_emb * pos_item_emb).sum(dim=1)
        neg_scores = (user_emb * neg_item_emb).sum(dim=1)
        
        return -F.logsigmoid(pos_scores - neg_scores).mean()
    
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
        scores = self.forward(users)
        
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
    
    def get_embeddings(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Get user and item embeddings."""
        return self.user_embedding.weight, self.item_embedding.weight
    
    def save(self, path: Path) -> None:
        """Save model."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "user_items": self._user_items,
            "config": {
                "n_factors": self.n_factors,
                "n_epochs": self.n_epochs,
                "lr": self.lr,
                "reg": self.reg,
                "batch_size": self.batch_size,
            },
        }, path)
    
    def load(self, path: Path) -> None:
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self._user_items = checkpoint.get("user_items")
    
    def __repr__(self) -> str:
        return f"SVDRecommender(n_factors={self.n_factors}, loss=BPR, device={self.device})"
