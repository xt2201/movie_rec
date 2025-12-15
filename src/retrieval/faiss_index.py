"""
FAISS-based retrieval for efficient similarity search.

Supports multiple index types:
- Flat: Exact search (small datasets)
- IVFFlat: Inverted file with flat quantizer (balanced)
- IVFPQ: Product quantization (large scale)
- HNSW: Hierarchical navigable small world (real-time)
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class IndexType(str, Enum):
    """Supported FAISS index types."""
    FLAT = "Flat"
    IVF_FLAT = "IVFFlat"
    IVF_PQ = "IVFPQ"
    HNSW = "HNSW"


class FAISSIndex:
    """
    FAISS index wrapper for efficient nearest neighbor search.
    
    Supports both CPU and GPU execution.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: Union[str, IndexType] = IndexType.IVF_FLAT,
        metric: str = "inner_product",
        nlist: int = 100,
        nprobe: int = 10,
        use_gpu: bool = False,
        m_pq: int = 8,
        nbits_pq: int = 8,
        m_hnsw: int = 32,
    ):
        """
        Initialize FAISS index.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index
            metric: Distance metric ('inner_product', 'l2')
            nlist: Number of clusters for IVF indexes
            nprobe: Number of clusters to search
            use_gpu: Whether to use GPU
            m_pq: Number of subquantizers for PQ
            nbits_pq: Bits per subquantizer
            m_hnsw: Number of connections per layer for HNSW
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu or faiss-gpu")
        
        self.embedding_dim = embedding_dim
        self.index_type = IndexType(index_type) if isinstance(index_type, str) else index_type
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu and self._check_gpu()
        self.m_pq = m_pq
        self.nbits_pq = nbits_pq
        self.m_hnsw = m_hnsw
        
        self.index: Optional[faiss.Index] = None
        self.is_trained = False
        self.gpu_resources: Optional[faiss.StandardGpuResources] = None
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available for FAISS."""
        try:
            return faiss.get_num_gpus() > 0
        except:
            return False
    
    def _get_metric(self) -> int:
        """Get FAISS metric type."""
        if self.metric == "inner_product":
            return faiss.METRIC_INNER_PRODUCT
        elif self.metric == "l2":
            return faiss.METRIC_L2
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def build(self, embeddings: Union[np.ndarray, torch.Tensor]) -> "FAISSIndex":
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Embeddings of shape (n_items, embedding_dim)
            
        Returns:
            Self for chaining
        """
        # Convert to numpy float32
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        embeddings = embeddings.astype(np.float32)
        
        # Normalize for inner product (cosine similarity)
        if self.metric == "inner_product":
            faiss.normalize_L2(embeddings)
        
        n_items, dim = embeddings.shape
        assert dim == self.embedding_dim, f"Embedding dim mismatch: {dim} vs {self.embedding_dim}"
        
        metric = self._get_metric()
        
        # Create index based on type
        if self.index_type == IndexType.FLAT:
            self.index = faiss.IndexFlatIP(dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
            
        elif self.index_type == IndexType.IVF_FLAT:
            quantizer = faiss.IndexFlatIP(dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, min(self.nlist, n_items), metric)
            
        elif self.index_type == IndexType.IVF_PQ:
            quantizer = faiss.IndexFlatIP(dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFPQ(quantizer, dim, min(self.nlist, n_items), self.m_pq, self.nbits_pq)
            
        elif self.index_type == IndexType.HNSW:
            self.index = faiss.IndexHNSWFlat(dim, self.m_hnsw, metric)
        
        # Train if needed (IVF indexes)
        if hasattr(self.index, "train") and not self.index.is_trained:
            self.index.train(embeddings)
        
        # Add vectors
        self.index.add(embeddings)
        self.is_trained = True
        
        # Set nprobe for IVF indexes
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe
        
        # Move to GPU if requested
        if self.use_gpu:
            self._to_gpu()
        
        return self
    
    def _to_gpu(self) -> None:
        """Move index to GPU."""
        if self.gpu_resources is None:
            self.gpu_resources = faiss.StandardGpuResources()
        
        self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
    
    def _to_cpu(self) -> None:
        """Move index back to CPU."""
        if self.use_gpu and self.index is not None:
            self.index = faiss.index_gpu_to_cpu(self.index)
    
    def search(
        self,
        queries: Union[np.ndarray, torch.Tensor],
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            queries: Query embeddings of shape (n_queries, embedding_dim)
            k: Number of neighbors to return
            
        Returns:
            (distances, indices) each of shape (n_queries, k)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")
        
        # Convert to numpy float32
        if isinstance(queries, torch.Tensor):
            queries = queries.detach().cpu().numpy()
        queries = queries.astype(np.float32)
        
        # Normalize for inner product
        if self.metric == "inner_product":
            faiss.normalize_L2(queries)
        
        distances, indices = self.index.search(queries, k)
        
        return distances, indices
    
    def search_torch(
        self,
        queries: torch.Tensor,
        k: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Search and return PyTorch tensors.
        
        Args:
            queries: Query embeddings
            k: Number of neighbors
            
        Returns:
            (distances, indices) as PyTorch tensors
        """
        distances, indices = self.search(queries, k)
        return (
            torch.from_numpy(distances),
            torch.from_numpy(indices).long(),
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save index to disk."""
        if self.index is None:
            raise RuntimeError("No index to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move to CPU before saving
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(path))
        else:
            faiss.write_index(self.index, str(path))
    
    def load(self, path: Union[str, Path]) -> "FAISSIndex":
        """Load index from disk."""
        path = Path(path)
        self.index = faiss.read_index(str(path))
        self.is_trained = True
        
        if self.use_gpu:
            self._to_gpu()
        
        return self
    
    @property
    def ntotal(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal if self.index is not None else 0
    
    def __repr__(self) -> str:
        return (
            f"FAISSIndex(type={self.index_type.value}, "
            f"dim={self.embedding_dim}, "
            f"ntotal={self.ntotal}, "
            f"gpu={self.use_gpu})"
        )


class FAISSRetriever:
    """
    High-level retriever combining model embeddings with FAISS search.
    
    Designed for recommendation systems where we need to:
    1. Get user embeddings from a model
    2. Search item embeddings via FAISS
    3. Filter out already-seen items
    """
    
    def __init__(
        self,
        model,
        index_type: Union[str, IndexType] = IndexType.IVF_FLAT,
        use_gpu: bool = False,
        **faiss_kwargs,
    ):
        """
        Initialize retriever.
        
        Args:
            model: Recommendation model with get_item_embeddings() method
            index_type: FAISS index type
            use_gpu: Whether to use GPU
            **faiss_kwargs: Additional arguments for FAISSIndex
        """
        self.model = model
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.faiss_kwargs = faiss_kwargs
        
        self.index: Optional[FAISSIndex] = None
    
    def build_index(self) -> "FAISSRetriever":
        """Build FAISS index from model's item embeddings."""
        self.model.eval()
        with torch.no_grad():
            item_embeddings = self.model.get_item_embeddings()
        
        embedding_dim = item_embeddings.shape[1]
        
        self.index = FAISSIndex(
            embedding_dim=embedding_dim,
            index_type=self.index_type,
            use_gpu=self.use_gpu,
            **self.faiss_kwargs,
        )
        
        self.index.build(item_embeddings)
        
        return self
    
    def retrieve(
        self,
        user_ids: Union[np.ndarray, torch.Tensor, list],
        k: int = 10,
        exclude_items: Optional[dict[int, set[int]]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k items for users.
        
        Args:
            user_ids: User indices
            k: Number of items to retrieve
            exclude_items: Dict mapping user_idx -> set of item_idx to exclude
            **kwargs: Additional arguments for model.get_user_embeddings()
            
        Returns:
            (scores, item_indices) of shape (n_users, k)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Convert user_ids to tensor
        if isinstance(user_ids, list):
            user_ids = torch.tensor(user_ids, dtype=torch.long)
        elif isinstance(user_ids, np.ndarray):
            user_ids = torch.from_numpy(user_ids).long()
        
        # Get user embeddings
        self.model.eval()
        with torch.no_grad():
            user_embeddings = self.model.get_user_embeddings(user_ids, **kwargs)
        
        # Retrieve more candidates if we need to filter
        retrieve_k = k * 2 if exclude_items else k
        
        # FAISS search
        distances, indices = self.index.search_torch(user_embeddings, retrieve_k)
        
        # Filter excluded items
        if exclude_items:
            filtered_scores = []
            filtered_indices = []
            
            for i, user_id in enumerate(user_ids.tolist()):
                user_exclude = exclude_items.get(user_id, set())
                
                user_scores = []
                user_items = []
                
                for score, item_idx in zip(distances[i], indices[i]):
                    if item_idx.item() not in user_exclude:
                        user_scores.append(score)
                        user_items.append(item_idx)
                        
                        if len(user_items) >= k:
                            break
                
                # Pad if needed
                while len(user_items) < k:
                    user_scores.append(torch.tensor(float("-inf")))
                    user_items.append(torch.tensor(-1))
                
                filtered_scores.append(torch.stack(user_scores[:k]))
                filtered_indices.append(torch.stack(user_items[:k]))
            
            distances = torch.stack(filtered_scores)
            indices = torch.stack(filtered_indices)
        else:
            distances = distances[:, :k]
            indices = indices[:, :k]
        
        return distances, indices
    
    def save(self, path: Union[str, Path]) -> None:
        """Save index to disk."""
        if self.index is not None:
            self.index.save(path)
    
    def load(self, path: Union[str, Path]) -> "FAISSRetriever":
        """Load index from disk."""
        if self.index is None:
            # Need to determine embedding dim from model
            self.model.eval()
            with torch.no_grad():
                item_embeddings = self.model.get_item_embeddings()
            embedding_dim = item_embeddings.shape[1]
            
            self.index = FAISSIndex(
                embedding_dim=embedding_dim,
                index_type=self.index_type,
                use_gpu=self.use_gpu,
                **self.faiss_kwargs,
            )
        
        self.index.load(path)
        return self
