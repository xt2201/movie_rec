"""
Base recommender abstract class defining the interface for all models.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn


class BaseRecommender(ABC, nn.Module):
    """
    Abstract base class for all recommendation models.
    
    All recommender models should inherit from this class and implement
    the required abstract methods for a unified interface.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize base recommender.
        
        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            embedding_dim: Dimension of embeddings
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
            **kwargs: Additional model-specific arguments
        """
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # Auto-detect device
        if device is None or device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)
    
    @property
    def device(self) -> torch.device:
        """Get the device this model is on."""
        return self._device
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.
        
        The exact signature depends on the model type:
        - GNN models: (edge_index, edge_weight) -> (user_emb, item_emb)
        - NCF models: (user_ids, item_ids) -> scores
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            users: User indices tensor
            pos_items: Positive item indices tensor
            neg_items: Negative item indices tensor
            **kwargs: Additional arguments (e.g., edge_index for GNN)
            
        Returns:
            Loss tensor
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        user_ids: torch.Tensor,
        item_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predict scores for user-item pairs.
        
        Args:
            user_ids: User indices tensor
            item_ids: Item indices tensor (if None, predict for all items)
            **kwargs: Additional arguments
            
        Returns:
            Score tensor of shape (batch_size,) or (batch_size, num_items)
        """
        pass
    
    @abstractmethod
    def recommend(
        self,
        user_ids: torch.Tensor,
        k: int = 10,
        exclude_items: Optional[dict[int, set[int]]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k recommendations for users.
        
        Args:
            user_ids: User indices tensor
            k: Number of recommendations
            exclude_items: Dict mapping user_idx -> set of item_idx to exclude
            **kwargs: Additional arguments
            
        Returns:
            (scores, item_indices) of shape (batch_size, k)
        """
        pass
    
    def get_user_embeddings(self, user_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get user embeddings.
        
        Args:
            user_ids: User indices (if None, return all embeddings)
            
        Returns:
            Embedding tensor of shape (num_users, embedding_dim) or (batch_size, embedding_dim)
        """
        raise NotImplementedError("Subclass must implement get_user_embeddings")
    
    def get_item_embeddings(self, item_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get item embeddings.
        
        Args:
            item_ids: Item indices (if None, return all embeddings)
            
        Returns:
            Embedding tensor of shape (num_items, embedding_dim) or (batch_size, embedding_dim)
        """
        raise NotImplementedError("Subclass must implement get_item_embeddings")
    
    def save(self, path: str | Path, **kwargs) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
            **kwargs: Additional metadata to save
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "num_users": self.num_users,
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim,
            "model_class": self.__class__.__name__,
            **kwargs,
        }
        
        torch.save(checkpoint, path)
    
    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "BaseRecommender":
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
            **kwargs: Override saved parameters
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location="cpu")
        
        # Get model parameters from checkpoint
        num_users = kwargs.get("num_users", checkpoint["num_users"])
        num_items = kwargs.get("num_items", checkpoint["num_items"])
        embedding_dim = kwargs.get("embedding_dim", checkpoint["embedding_dim"])
        
        # Create model instance
        model = cls(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            **kwargs,
        )
        
        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model
    
    def freeze(self) -> None:
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_users={self.num_users}, "
            f"num_items={self.num_items}, "
            f"embedding_dim={self.embedding_dim}, "
            f"params={self.count_parameters():,})"
        )
