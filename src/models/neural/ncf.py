"""
NCF: Neural Collaborative Filtering

Paper: https://arxiv.org/abs/1708.05031
Reference: NVIDIA DeepLearningExamples/PyTorch/Recommendation/NCF

Implements:
- GMF: Generalized Matrix Factorization
- MLP: Multi-Layer Perceptron  
- NeuMF: Neural Matrix Factorization (GMF + MLP fusion)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseRecommender


class GMF(nn.Module):
    """
    Generalized Matrix Factorization.
    
    Learns user-item interaction as element-wise product:
    y = sigmoid(h^T (p_u ⊙ q_i))
    """
    
    def __init__(self, num_users: int, num_items: int, latent_dim: int):
        super().__init__()
        
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        self.output_layer = nn.Linear(latent_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.kaiming_uniform_(self.output_layer.weight, a=1)
    
    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        
        # Element-wise product
        x = user_emb * item_emb
        
        return self.output_layer(x)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for collaborative filtering.
    
    Learns user-item interaction through concatenation + FC layers:
    y = sigmoid(h^T ReLU(W^L(...ReLU(W^1 [p_u; q_i])...)))
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        layer_sizes: list[int],
        dropout: float = 0.0,
    ):
        super().__init__()
        
        if layer_sizes[0] % 2 != 0:
            raise ValueError("First layer size must be even for user/item embedding concat")
        
        embed_dim = layer_sizes[0] // 2
        
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        
        # MLP layers
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        
        self.output_layer = nn.Linear(layer_sizes[-1], 1)
        self.dropout = dropout
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        for layer in self.layers:
            # Glorot uniform initialization
            fan_in, fan_out = layer.in_features, layer.out_features
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            nn.init.uniform_(layer.weight, -limit, limit)
            nn.init.zeros_(layer.bias)
    
    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        
        # Concatenation
        x = torch.cat([user_emb, item_emb], dim=-1)
        
        # MLP layers
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return self.output_layer(x)


class NeuMF(nn.Module):
    """
    Neural Matrix Factorization.
    
    Combines GMF and MLP:
    y = sigmoid(h^T [GMF_output; MLP_output])
    
    Uses separate embeddings for GMF and MLP branches.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        mf_dim: int = 64,
        mlp_layer_sizes: list[int] = [128, 64, 32, 16],
        dropout: float = 0.0,
    ):
        super().__init__()
        
        if mlp_layer_sizes[0] % 2 != 0:
            raise ValueError("First MLP layer size must be even")
        
        mlp_embed_dim = mlp_layer_sizes[0] // 2
        
        # GMF embeddings
        self.mf_user_embedding = nn.Embedding(num_users, mf_dim)
        self.mf_item_embedding = nn.Embedding(num_items, mf_dim)
        
        # MLP embeddings (separate from GMF)
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_embed_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_embed_dim)
        
        # MLP layers
        self.mlp_layers = nn.ModuleList()
        for i in range(len(mlp_layer_sizes) - 1):
            self.mlp_layers.append(nn.Linear(mlp_layer_sizes[i], mlp_layer_sizes[i + 1]))
        
        # Final prediction layer (concatenates GMF and MLP outputs)
        self.output_layer = nn.Linear(mf_dim + mlp_layer_sizes[-1], 1)
        
        self.dropout = dropout
        self.mf_dim = mf_dim
        self.mlp_dim = mlp_layer_sizes[-1]
        
        self._init_weights()
    
    def _init_weights(self):
        # Normal initialization for embeddings
        nn.init.normal_(self.mf_user_embedding.weight, std=0.01)
        nn.init.normal_(self.mf_item_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_user_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embedding.weight, std=0.01)
        
        # Glorot uniform for MLP layers
        for layer in self.mlp_layers:
            fan_in, fan_out = layer.in_features, layer.out_features
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            nn.init.uniform_(layer.weight, -limit, limit)
            nn.init.zeros_(layer.bias)
        
        # LeCun uniform for final layer
        fan_in = self.output_layer.in_features
        limit = np.sqrt(3.0 / fan_in)
        nn.init.uniform_(self.output_layer.weight, -limit, limit)
    
    def forward(
        self,
        user: torch.Tensor,
        item: torch.Tensor,
        sigmoid: bool = False,
    ) -> torch.Tensor:
        # GMF branch
        mf_user = self.mf_user_embedding(user)
        mf_item = self.mf_item_embedding(item)
        mf_output = mf_user * mf_item  # Element-wise product
        
        # MLP branch
        mlp_user = self.mlp_user_embedding(user)
        mlp_item = self.mlp_item_embedding(item)
        mlp_output = torch.cat([mlp_user, mlp_item], dim=-1)
        
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)
            mlp_output = F.relu(mlp_output)
            if self.dropout > 0:
                mlp_output = F.dropout(mlp_output, p=self.dropout, training=self.training)
        
        # Concatenate GMF and MLP
        concat = torch.cat([mf_output, mlp_output], dim=-1)
        output = self.output_layer(concat)
        
        if sigmoid:
            output = torch.sigmoid(output)
        
        return output
    
    def get_user_embedding(self, user: torch.Tensor) -> torch.Tensor:
        """Get combined user embedding for FAISS indexing."""
        mf_emb = self.mf_user_embedding(user)
        mlp_emb = self.mlp_user_embedding(user)
        return torch.cat([mf_emb, mlp_emb], dim=-1)
    
    def get_item_embedding(self, item: torch.Tensor) -> torch.Tensor:
        """Get combined item embedding for FAISS indexing."""
        mf_emb = self.mf_item_embedding(item)
        mlp_emb = self.mlp_item_embedding(item)
        return torch.cat([mf_emb, mlp_emb], dim=-1)


class NCF(BaseRecommender):
    """
    Neural Collaborative Filtering wrapper implementing BaseRecommender interface.
    
    Supports three modes:
    - 'gmf': GMF only
    - 'mlp': MLP only  
    - 'neumf': Full NeuMF (GMF + MLP fusion)
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        mf_dim: Optional[int] = None,
        mlp_layers: list[int] = [128, 64, 32, 16],
        dropout: float = 0.2,
        mode: str = "neumf",
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize NCF model.
        
        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            embedding_dim: Base embedding dimension (used if mf_dim not specified)
            mf_dim: GMF embedding dimension
            mlp_layers: MLP layer sizes
            dropout: Dropout rate
            mode: Model mode ('gmf', 'mlp', 'neumf')
            device: Device to use
        """
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            device=device,
            **kwargs,
        )
        
        self.mode = mode
        self.mf_dim = mf_dim or embedding_dim
        self.mlp_layers = mlp_layers
        self.dropout_rate = dropout
        
        # Build model
        if mode == "gmf":
            self.model = GMF(num_users, num_items, self.mf_dim)
        elif mode == "mlp":
            self.model = MLP(num_users, num_items, mlp_layers, dropout)
        elif mode == "neumf":
            self.model = NeuMF(num_users, num_items, self.mf_dim, mlp_layers, dropout)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'gmf', 'mlp', or 'neumf'")
    
    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        return self.model(user, item)
    
    def compute_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute BCE loss for NCF.
        
        Uses BCEWithLogitsLoss for numerical stability.
        Compatible with BPR-style trainer (user, pos_item, neg_item).
        """
        # Ensure indices are Long type for embedding lookup
        users = users.long()
        pos_items = pos_items.long()
        neg_items = neg_items.long()
        
        # Positive samples
        pos_logits = self.model(users, pos_items).squeeze()
        pos_labels = torch.ones_like(pos_logits)
        
        # Negative samples
        neg_logits = self.model(users, neg_items).squeeze()
        neg_labels = torch.zeros_like(neg_logits)
        
        # Combine
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([pos_labels, neg_labels])
        
        return F.binary_cross_entropy_with_logits(logits, labels)
    
    def predict(
        self,
        user_ids: torch.Tensor,
        item_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predict scores for user-item pairs.
        
        If item_ids is None, predicts for all items (expensive).
        """
        self.eval()
        with torch.no_grad():
            if item_ids is not None:
                return torch.sigmoid(self.model(user_ids, item_ids).squeeze())
            
            # Predict for all items
            batch_size = len(user_ids)
            all_items = torch.arange(self.num_items, device=user_ids.device)
            
            scores = []
            for user in user_ids:
                user_expanded = user.expand(self.num_items)
                score = torch.sigmoid(self.model(user_expanded, all_items).squeeze())
                scores.append(score)
            
            return torch.stack(scores)
    
    def recommend(
        self,
        user_ids: torch.Tensor,
        k: int = 10,
        exclude_items: Optional[dict[int, set[int]]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get top-k recommendations."""
        scores = self.predict(user_ids)
        
        if exclude_items is not None:
            for i, user_id in enumerate(user_ids.tolist()):
                if user_id in exclude_items:
                    for item_id in exclude_items[user_id]:
                        if item_id < self.num_items:
                            scores[i, item_id] = float("-inf")
        
        top_scores, top_items = torch.topk(scores, k, dim=1)
        return top_scores, top_items
    
    def get_user_embeddings(self, user_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get user embeddings for FAISS indexing."""
        if self.mode == "neumf":
            if user_ids is None:
                user_ids = torch.arange(self.num_users, device=self.device)
            return self.model.get_user_embedding(user_ids)
        elif self.mode == "gmf":
            if user_ids is None:
                return self.model.user_embedding.weight
            return self.model.user_embedding(user_ids)
        else:  # mlp
            if user_ids is None:
                return self.model.user_embedding.weight
            return self.model.user_embedding(user_ids)
    
    def get_item_embeddings(self, item_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get item embeddings for FAISS indexing."""
        if self.mode == "neumf":
            if item_ids is None:
                item_ids = torch.arange(self.num_items, device=self.device)
            return self.model.get_item_embedding(item_ids)
        elif self.mode == "gmf":
            if item_ids is None:
                return self.model.item_embedding.weight
            return self.model.item_embedding(item_ids)
        else:  # mlp
            if item_ids is None:
                return self.model.item_embedding.weight
            return self.model.item_embedding(item_ids)
    
    def load_pretrained(self, gmf_path: Optional[str], mlp_path: Optional[str]) -> None:
        """
        Load pretrained GMF and MLP weights for NeuMF.
        
        This is the recommended training procedure:
        1. Pretrain GMF and MLP separately
        2. Initialize NeuMF with pretrained weights
        3. Fine-tune the full model
        """
        if self.mode != "neumf":
            raise ValueError("Pretrained loading only supported for NeuMF mode")
        
        if gmf_path:
            gmf_state = torch.load(gmf_path, map_location="cpu")
            self.model.mf_user_embedding.weight.data.copy_(
                gmf_state["model.user_embedding.weight"]
            )
            self.model.mf_item_embedding.weight.data.copy_(
                gmf_state["model.item_embedding.weight"]
            )
        
        if mlp_path:
            mlp_state = torch.load(mlp_path, map_location="cpu")
            self.model.mlp_user_embedding.weight.data.copy_(
                mlp_state["model.user_embedding.weight"]
            )
            self.model.mlp_item_embedding.weight.data.copy_(
                mlp_state["model.item_embedding.weight"]
            )
            
            for i, layer in enumerate(self.model.mlp_layers):
                layer.weight.data.copy_(mlp_state[f"model.layers.{i}.weight"])
                layer.bias.data.copy_(mlp_state[f"model.layers.{i}.bias"])
