"""
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

Paper: https://arxiv.org/abs/2002.02126
Reference: Original implementation from Repo A
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree

from ..base import BaseRecommender


class LightGCNConv(MessagePassing):
    """
    LightGCN Convolution Layer.
    
    Simplified graph convolution that only aggregates neighbor embeddings
    without feature transformation or nonlinear activation.
    
    Message: m_{j->i} = e_j / sqrt(deg(i) * deg(j))
    Aggregation: e_i^{(l+1)} = sum_{j in N(i)} m_{j->i}
    """
    
    def __init__(self, normalize: bool = True, **kwargs):
        """
        Initialize LightGCN convolution.
        
        Args:
            normalize: Whether to apply symmetric normalization
        """
        super().__init__(aggr="add", **kwargs)
        self.normalize = normalize
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node embeddings of shape (num_nodes, embedding_dim)
            edge_index: Edge index of shape (2, num_edges)
            edge_weight: Optional precomputed edge weights
            
        Returns:
            Updated node embeddings of shape (num_nodes, embedding_dim)
        """
        if edge_weight is None and self.normalize:
            # Compute symmetric normalization weights
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
    
    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        """Construct messages from source nodes."""
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j
    
    def message_and_aggregate(self, adj_t, x):
        """Optimized message passing for sparse tensor."""
        return adj_t.matmul(x, reduce=self.aggr)


class LightGCN(BaseRecommender):
    """
    LightGCN Model for Collaborative Filtering.
    
    Key ideas:
    1. Remove feature transformation and nonlinear activation from GCN
    2. Aggregate embeddings from different layers via mean pooling
    3. Use BPR loss for implicit feedback
    
    Architecture:
    - Embedding layer for users and items
    - Multiple LightGCN convolution layers
    - Layer combination via mean pooling
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
        normalize: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize LightGCN model.
        
        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            embedding_dim: Dimension of embeddings
            num_layers: Number of graph convolution layers
            dropout: Dropout rate (applied to initial embeddings)
            normalize: Whether to use symmetric normalization
            device: Device to use
        """
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            device=device,
            **kwargs,
        )
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.normalize = normalize
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # LightGCN convolution layers (shared parameters)
        self.convs = nn.ModuleList([
            LightGCNConv(normalize=normalize) for _ in range(num_layers)
        ])
        
        # Initialize embeddings
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with Xavier uniform."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def get_initial_embeddings(self) -> torch.Tensor:
        """Get concatenated user and item embeddings."""
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        return torch.cat([user_emb, item_emb], dim=0)
    
    def forward(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LightGCN layers.
        
        Args:
            edge_index: Edge index of shape (2, num_edges)
            edge_weight: Optional precomputed edge weights
            
        Returns:
            (user_embeddings, item_embeddings) after layer aggregation
        """
        # Get initial embeddings
        x = self.get_initial_embeddings()
        
        # Apply dropout to initial embeddings
        if self.dropout > 0 and self.training:
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Store embeddings from each layer
        all_embeddings = [x]
        
        # Message passing through layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            all_embeddings.append(x)
        
        # Layer combination via mean pooling
        all_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = all_embeddings.mean(dim=1)
        
        # Split back to users and items
        user_emb, item_emb = torch.split(
            final_embeddings, [self.num_users, self.num_items]
        )
        
        return user_emb, item_emb
    
    def compute_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        reg_weight: float = 1e-5,
    ) -> torch.Tensor:
        """
        Compute BPR loss with L2 regularization.
        
        Args:
            users: User indices
            pos_items: Positive item indices
            neg_items: Negative item indices
            edge_index: Graph edge index
            edge_weight: Optional edge weights
            reg_weight: L2 regularization weight
            
        Returns:
            Total loss
        """
        # Get embeddings from GCN
        user_emb, item_emb = self.forward(edge_index, edge_weight)
        
        # Get embeddings for batch
        user_emb_batch = user_emb[users]
        pos_item_emb = item_emb[pos_items]
        neg_item_emb = item_emb[neg_items]
        
        # BPR scores
        pos_scores = (user_emb_batch * pos_item_emb).sum(dim=1)
        neg_scores = (user_emb_batch * neg_item_emb).sum(dim=1)
        
        # BPR loss
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # L2 regularization on initial embeddings
        reg_loss = reg_weight * (
            self.user_embedding.weight[users].norm(2).pow(2) +
            self.item_embedding.weight[pos_items].norm(2).pow(2) +
            self.item_embedding.weight[neg_items].norm(2).pow(2)
        ) / len(users)
        
        return bpr_loss + reg_loss
    
    def predict(
        self,
        user_ids: torch.Tensor,
        item_ids: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        user_emb: Optional[torch.Tensor] = None,
        item_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict scores for user-item pairs.
        
        Args:
            user_ids: User indices
            item_ids: Item indices (if None, predict for all items)
            edge_index: Graph edge index (required if embeddings not provided)
            edge_weight: Optional edge weights
            user_emb: Precomputed user embeddings
            item_emb: Precomputed item embeddings
            
        Returns:
            Score tensor
        """
        # Get embeddings if not provided
        if user_emb is None or item_emb is None:
            if edge_index is None:
                raise ValueError("Either provide embeddings or edge_index")
            user_emb, item_emb = self.forward(edge_index, edge_weight)
        
        user_emb_batch = user_emb[user_ids]
        
        if item_ids is None:
            # Score all items
            scores = torch.matmul(user_emb_batch, item_emb.t())
        else:
            # Score specific items
            item_emb_batch = item_emb[item_ids]
            scores = (user_emb_batch * item_emb_batch).sum(dim=1)
        
        return scores
    
    def recommend(
        self,
        user_ids: torch.Tensor,
        k: int = 10,
        exclude_items: Optional[dict[int, set[int]]] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        user_emb: Optional[torch.Tensor] = None,
        item_emb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k recommendations for users.
        
        Args:
            user_ids: User indices
            k: Number of recommendations
            exclude_items: Dict mapping user_idx -> set of item_idx to exclude
            edge_index: Graph edge index
            edge_weight: Optional edge weights
            user_emb: Precomputed user embeddings
            item_emb: Precomputed item embeddings
            
        Returns:
            (scores, item_indices) of shape (batch_size, k)
        """
        # Get all scores
        scores = self.predict(
            user_ids=user_ids,
            item_ids=None,
            edge_index=edge_index,
            edge_weight=edge_weight,
            user_emb=user_emb,
            item_emb=item_emb,
        )
        
        # Mask excluded items
        if exclude_items is not None:
            for i, user_id in enumerate(user_ids.tolist()):
                if user_id in exclude_items:
                    for item_id in exclude_items[user_id]:
                        if item_id < self.num_items:
                            scores[i, item_id] = float("-inf")
        
        # Get top-k
        top_scores, top_items = torch.topk(scores, k, dim=1)
        
        return top_scores, top_items
    
    def get_user_embeddings(
        self,
        user_ids: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get user embeddings after GCN propagation."""
        if edge_index is None:
            # Return initial embeddings
            if user_ids is None:
                return self.user_embedding.weight
            return self.user_embedding(user_ids)
        
        user_emb, _ = self.forward(edge_index, edge_weight)
        
        if user_ids is None:
            return user_emb
        return user_emb[user_ids]
    
    def get_item_embeddings(
        self,
        item_ids: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get item embeddings after GCN propagation."""
        if edge_index is None:
            # Return initial embeddings
            if item_ids is None:
                return self.item_embedding.weight
            return self.item_embedding(item_ids)
        
        _, item_emb = self.forward(edge_index, edge_weight)
        
        if item_ids is None:
            return item_emb
        return item_emb[item_ids]
