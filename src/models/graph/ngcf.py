"""
NGCF: Neural Graph Collaborative Filtering

Paper: https://arxiv.org/abs/1905.08108
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


class NGCFConv(MessagePassing):
    """
    NGCF Convolution Layer.
    
    Unlike LightGCN, NGCF includes:
    1. Feature transformation (linear layers)
    2. Nonlinear activation (LeakyReLU)
    3. Message dropout
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        """
        Initialize NGCF convolution.
        
        Args:
            in_channels: Input embedding dimension
            out_channels: Output embedding dimension
            dropout: Message dropout rate
        """
        super().__init__(aggr="add", **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        
        # Transformation matrices
        self.W1 = nn.Linear(in_channels, out_channels, bias=True)
        self.W2 = nn.Linear(in_channels, out_channels, bias=True)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.zeros_(self.W1.bias)
        nn.init.zeros_(self.W2.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node embeddings of shape (num_nodes, in_channels)
            edge_index: Edge index of shape (2, num_edges)
            edge_weight: Optional precomputed edge weights
            
        Returns:
            Updated node embeddings of shape (num_nodes, out_channels)
        """
        if edge_weight is None:
            # Compute symmetric normalization weights
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Self-connection term: W1 * e_i
        self_term = self.W1(x)
        
        # Neighbor aggregation term
        neighbor_term = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
        # Combine
        out = self_term + neighbor_term
        
        # Activation
        out = F.leaky_relu(out, negative_slope=0.2)
        
        # Dropout
        if self.dropout > 0 and self.training:
            out = F.dropout(out, p=self.dropout, training=self.training)
        
        return out
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Construct messages.
        
        m_{j->i} = (W1 * e_j + W2 * (e_j ⊙ e_i)) / sqrt(|N_i| * |N_j|)
        """
        # Element-wise interaction
        interaction = x_j * x_i  # e_j ⊙ e_i
        
        # Transform
        msg = self.W1(x_j) + self.W2(interaction)
        
        # Normalize
        return edge_weight.view(-1, 1) * msg


class NGCF(BaseRecommender):
    """
    Neural Graph Collaborative Filtering Model.
    
    Key differences from LightGCN:
    1. Feature transformation in each layer
    2. Element-wise product interaction
    3. Concatenation of layer embeddings (instead of mean)
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dims: list[int] = [64, 64, 64],
        dropout: float = 0.1,
        message_dropout: float = 0.1,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize NGCF model.
        
        Args:
            num_users: Number of unique users
            num_items: Number of unique items  
            embedding_dim: Initial embedding dimension
            hidden_dims: List of hidden dimensions for each layer
            dropout: Embedding dropout
            message_dropout: Message passing dropout
            device: Device to use
        """
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            device=device,
            **kwargs,
        )
        
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.message_dropout = message_dropout
        
        # Total embedding dim = initial + sum of hidden dims (due to concatenation)
        self.total_embedding_dim = embedding_dim + sum(hidden_dims)
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # NGCF convolution layers
        self.convs = nn.ModuleList()
        in_dim = embedding_dim
        for out_dim in hidden_dims:
            self.convs.append(NGCFConv(in_dim, out_dim, dropout=message_dropout))
            in_dim = out_dim
        
        self._init_weights()
    
    def _init_weights(self):
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
        Forward pass through NGCF layers.
        
        Args:
            edge_index: Edge index of shape (2, num_edges)
            edge_weight: Optional precomputed edge weights
            
        Returns:
            (user_embeddings, item_embeddings) after layer concatenation
        """
        # Get initial embeddings
        x = self.get_initial_embeddings()
        
        # Apply dropout
        if self.dropout > 0 and self.training:
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Store embeddings from each layer
        all_embeddings = [x]
        
        # Message passing
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            all_embeddings.append(x)
        
        # Layer combination via concatenation
        final_embeddings = torch.cat(all_embeddings, dim=1)
        
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
        """Compute BPR loss with L2 regularization."""
        # Get embeddings
        user_emb, item_emb = self.forward(edge_index, edge_weight)
        
        # Get batch embeddings
        user_emb_batch = user_emb[users]
        pos_item_emb = item_emb[pos_items]
        neg_item_emb = item_emb[neg_items]
        
        # BPR scores
        pos_scores = (user_emb_batch * pos_item_emb).sum(dim=1)
        neg_scores = (user_emb_batch * neg_item_emb).sum(dim=1)
        
        # BPR loss
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # L2 regularization
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
        """Predict scores for user-item pairs."""
        if user_emb is None or item_emb is None:
            if edge_index is None:
                raise ValueError("Either provide embeddings or edge_index")
            user_emb, item_emb = self.forward(edge_index, edge_weight)
        
        user_emb_batch = user_emb[user_ids]
        
        if item_ids is None:
            scores = torch.matmul(user_emb_batch, item_emb.t())
        else:
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
        """Get top-k recommendations."""
        scores = self.predict(
            user_ids=user_ids,
            item_ids=None,
            edge_index=edge_index,
            edge_weight=edge_weight,
            user_emb=user_emb,
            item_emb=item_emb,
        )
        
        if exclude_items is not None:
            for i, user_id in enumerate(user_ids.tolist()):
                if user_id in exclude_items:
                    for item_id in exclude_items[user_id]:
                        if item_id < self.num_items:
                            scores[i, item_id] = float("-inf")
        
        top_scores, top_items = torch.topk(scores, k, dim=1)
        return top_scores, top_items
    
    def get_user_embeddings(
        self,
        user_ids: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get user embeddings."""
        if edge_index is None:
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
        """Get item embeddings."""
        if edge_index is None:
            if item_ids is None:
                return self.item_embedding.weight
            return self.item_embedding(item_ids)
        
        _, item_emb = self.forward(edge_index, edge_weight)
        if item_ids is None:
            return item_emb
        return item_emb[item_ids]
