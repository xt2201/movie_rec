"""
Negative sampling strategies for recommendation training.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class NegativeSampler:
    """Negative sampler for implicit feedback recommendation."""
    
    def __init__(
        self,
        num_items: int,
        user_item_set: dict[int, set[int]],
        num_negatives: int = 4,
        strategy: str = "uniform",
        item_popularity: Optional[np.ndarray] = None,
    ):
        """
        Initialize negative sampler.
        
        Args:
            num_items: Total number of items
            user_item_set: Dict mapping user_idx -> set of positive item_idx
            num_negatives: Number of negative samples per positive
            strategy: Sampling strategy ('uniform', 'popularity')
            item_popularity: Item popularity counts (for popularity sampling)
        """
        self.num_items = num_items
        self.user_item_set = user_item_set
        self.num_negatives = num_negatives
        self.strategy = strategy
        
        # Precompute popularity distribution
        if strategy == "popularity" and item_popularity is not None:
            # P(item) ∝ popularity^0.75 (smoothed)
            popularity_smooth = np.power(item_popularity, 0.75)
            self.item_probs = popularity_smooth / popularity_smooth.sum()
        else:
            self.item_probs = None
    
    def sample(self, user_idx: int, num_samples: int = 1) -> list[int]:
        """
        Sample negative items for a user.
        
        Args:
            user_idx: User index
            num_samples: Number of negative samples
            
        Returns:
            List of negative item indices
        """
        positive_items = self.user_item_set.get(user_idx, set())
        negatives = []
        
        max_attempts = num_samples * 10
        attempts = 0
        
        while len(negatives) < num_samples and attempts < max_attempts:
            if self.strategy == "uniform":
                candidate = np.random.randint(0, self.num_items)
            else:  # popularity
                candidate = np.random.choice(self.num_items, p=self.item_probs)
            
            if candidate not in positive_items:
                negatives.append(candidate)
            attempts += 1
        
        # Fill remaining with random if needed
        while len(negatives) < num_samples:
            candidate = np.random.randint(0, self.num_items)
            negatives.append(candidate)
        
        return negatives
    
    def sample_batch(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample negatives for a batch of positive interactions.
        
        Args:
            user_indices: Array of user indices
            item_indices: Array of positive item indices
            
        Returns:
            (users, pos_items, neg_items) arrays
            Shape: users/pos_items repeat num_negatives times,
                   neg_items has num_negatives per positive
        """
        batch_size = len(user_indices)
        total_samples = batch_size * self.num_negatives
        
        users_expanded = np.repeat(user_indices, self.num_negatives)
        pos_items_expanded = np.repeat(item_indices, self.num_negatives)
        neg_items = np.zeros(total_samples, dtype=np.int64)
        
        for i, user_idx in enumerate(user_indices):
            negs = self.sample(user_idx, self.num_negatives)
            start_idx = i * self.num_negatives
            neg_items[start_idx:start_idx + self.num_negatives] = negs
        
        return users_expanded, pos_items_expanded, neg_items


class BPRDataset(Dataset):
    """Dataset for BPR (Bayesian Personalized Ranking) training."""
    
    def __init__(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        num_items: int,
        user_item_set: dict[int, set[int]],
        num_negatives: int = 1,
    ):
        """
        Initialize BPR dataset.
        
        Each sample returns (user, positive_item, negative_item).
        
        Args:
            user_indices: Array of user indices for positive interactions
            item_indices: Array of item indices for positive interactions
            num_items: Total number of items
            user_item_set: Dict mapping user_idx -> set of positive item_idx
            num_negatives: Number of negative samples per positive
        """
        self.user_indices = user_indices
        self.item_indices = item_indices
        self.num_items = num_items
        self.user_item_set = user_item_set
        self.num_negatives = num_negatives
        
        self.sampler = NegativeSampler(
            num_items=num_items,
            user_item_set=user_item_set,
            num_negatives=num_negatives,
        )
    
    def __len__(self) -> int:
        return len(self.user_indices)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user = self.user_indices[idx]
        pos_item = self.item_indices[idx]
        
        # Sample negative(s)
        neg_items = self.sampler.sample(user, self.num_negatives)
        
        if self.num_negatives == 1:
            return (
                torch.tensor(user, dtype=torch.long),
                torch.tensor(pos_item, dtype=torch.long),
                torch.tensor(neg_items[0], dtype=torch.long),
            )
        else:
            return (
                torch.tensor(user, dtype=torch.long),
                torch.tensor(pos_item, dtype=torch.long),
                torch.tensor(neg_items, dtype=torch.long),
            )


class NCFDataset(Dataset):
    """Dataset for NCF (Neural Collaborative Filtering) training."""
    
    def __init__(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        labels: np.ndarray,
        num_items: int,
        user_item_set: dict[int, set[int]],
        num_negatives: int = 4,
        mode: str = "train",
    ):
        """
        Initialize NCF dataset.
        
        For training: augments with negative samples (label=0).
        For evaluation: returns original interactions only.
        
        Args:
            user_indices: Array of user indices
            item_indices: Array of item indices
            labels: Array of labels (1 for positive, 0 for negative)
            num_items: Total number of items
            user_item_set: Dict mapping user_idx -> set of positive item_idx
            num_negatives: Number of negative samples per positive (training only)
            mode: 'train' or 'eval'
        """
        self.num_items = num_items
        self.user_item_set = user_item_set
        self.num_negatives = num_negatives
        self.mode = mode
        
        if mode == "train":
            # Augment with negative samples
            sampler = NegativeSampler(
                num_items=num_items,
                user_item_set=user_item_set,
                num_negatives=num_negatives,
            )
            
            all_users = []
            all_items = []
            all_labels = []
            
            for user, item in zip(user_indices, item_indices):
                # Positive sample
                all_users.append(user)
                all_items.append(item)
                all_labels.append(1.0)
                
                # Negative samples
                neg_items = sampler.sample(user, num_negatives)
                for neg_item in neg_items:
                    all_users.append(user)
                    all_items.append(neg_item)
                    all_labels.append(0.0)
            
            self.user_indices = np.array(all_users, dtype=np.int64)
            self.item_indices = np.array(all_items, dtype=np.int64)
            self.labels = np.array(all_labels, dtype=np.float32)
        else:
            self.user_indices = user_indices
            self.item_indices = item_indices
            self.labels = labels
    
    def __len__(self) -> int:
        return len(self.user_indices)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.user_indices[idx], dtype=torch.long),
            torch.tensor(self.item_indices[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )
