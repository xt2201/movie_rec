"""
Graph construction utilities for GNN-based recommendation models.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.sparse import csr_matrix


class GraphBuilder:
    """Builder for user-item bipartite graphs."""
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        undirected: bool = True,
        add_self_loops: bool = False,
    ):
        """
        Initialize graph builder.
        
        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            undirected: Whether to create undirected graph (add reverse edges)
            add_self_loops: Whether to add self-loops
        """
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.undirected = undirected
        self.add_self_loops = add_self_loops
        
    def build_edge_index(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
    ) -> torch.Tensor:
        """
        Build edge index for PyTorch Geometric.
        
        User nodes: 0 to num_users-1
        Item nodes: num_users to num_users+num_items-1
        
        Args:
            user_indices: Array of user indices
            item_indices: Array of item indices
            
        Returns:
            Edge index tensor of shape (2, num_edges)
        """
        # Shift item indices to be after user indices
        item_indices_shifted = item_indices + self.num_users
        
        # User -> Item edges
        src = user_indices
        dst = item_indices_shifted
        
        if self.undirected:
            # Add Item -> User edges (reverse)
            src = np.concatenate([src, dst])
            dst = np.concatenate([item_indices_shifted, user_indices])
            # Re-order: first half user->item, second half item->user
            src = np.concatenate([user_indices, item_indices_shifted])
            dst = np.concatenate([item_indices_shifted, user_indices])
        
        edge_index = torch.tensor(
            np.stack([src, dst], axis=0),
            dtype=torch.long,
        )
        
        if self.add_self_loops:
            self_loops = torch.arange(self.num_nodes, dtype=torch.long)
            self_loops = self_loops.unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_loops], dim=1)
        
        return edge_index
    
    def build_sparse_adjacency(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Build sparse adjacency matrix for message passing.
        
        Args:
            user_indices: Array of user indices
            item_indices: Array of item indices
            normalize: Whether to apply symmetric normalization (D^{-1/2} A D^{-1/2})
            
        Returns:
            Sparse adjacency tensor of shape (num_nodes, num_nodes)
        """
        # Build dense adjacency first (for simplicity with small datasets)
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        
        item_indices_shifted = item_indices + self.num_users
        
        # User -> Item
        adj[user_indices, item_indices_shifted] = 1.0
        
        if self.undirected:
            # Item -> User
            adj[item_indices_shifted, user_indices] = 1.0
        
        if self.add_self_loops:
            np.fill_diagonal(adj, 1.0)
        
        if normalize:
            # Symmetric normalization: D^{-1/2} A D^{-1/2}
            degree = np.sum(adj, axis=1)
            degree_inv_sqrt = np.power(degree, -0.5)
            degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
            
            # D^{-1/2} A
            adj = adj * degree_inv_sqrt.reshape(-1, 1)
            # D^{-1/2} A D^{-1/2}
            adj = adj * degree_inv_sqrt.reshape(1, -1)
        
        # Convert to sparse tensor
        adj_sparse = csr_matrix(adj)
        indices = torch.tensor(
            np.vstack([adj_sparse.row, adj_sparse.col]),
            dtype=torch.long,
        )
        values = torch.tensor(adj_sparse.data, dtype=torch.float32)
        
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=(self.num_nodes, self.num_nodes),
        ).coalesce()
    
    def compute_edge_weights(
        self,
        edge_index: torch.Tensor,
        normalization: str = "sym",
    ) -> torch.Tensor:
        """
        Compute edge weights for normalized message passing.
        
        Args:
            edge_index: Edge index tensor of shape (2, num_edges)
            normalization: Type of normalization ('sym', 'row', 'none')
            
        Returns:
            Edge weight tensor of shape (num_edges,)
        """
        if normalization == "none":
            return torch.ones(edge_index.size(1), dtype=torch.float32)
        
        # Compute degree
        src, dst = edge_index
        degree = torch.zeros(self.num_nodes, dtype=torch.float32)
        degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        
        if normalization == "sym":
            # 1 / sqrt(deg(src) * deg(dst))
            deg_src = degree[src]
            deg_dst = degree[dst]
            weights = 1.0 / torch.sqrt(deg_src * deg_dst)
        elif normalization == "row":
            # 1 / deg(src)
            deg_src = degree[src]
            weights = 1.0 / deg_src
        else:
            raise ValueError(f"Unknown normalization: {normalization}")
        
        # Handle division by zero
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        
        return weights
    
    def get_user_item_edge_index(
        self,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split edge index into user->item and item->user edges.
        
        Returns:
            (user_to_item_edges, item_to_user_edges) each of shape (2, num_edges/2)
        """
        src, dst = edge_index
        
        # User -> Item: src < num_users, dst >= num_users
        user_to_item_mask = (src < self.num_users) & (dst >= self.num_users)
        user_to_item = edge_index[:, user_to_item_mask]
        
        # Item -> User: src >= num_users, dst < num_users
        item_to_user_mask = (src >= self.num_users) & (dst < self.num_users)
        item_to_user = edge_index[:, item_to_user_mask]
        
        return user_to_item, item_to_user
