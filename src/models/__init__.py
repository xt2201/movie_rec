"""
Models package for recommendation approaches.

Includes:
- Graph Neural Networks (LightGCN, NGCF)
- Neural Collaborative Filtering (NCF, GMF, MLP)
- Traditional Methods (SVD, Item-CF)
- Hybrid Ensemble
"""
from .base import BaseRecommender
from .graph import LightGCN, NGCF
from .neural import NCF, GMF, MLP, NeuMF
from .traditional import SVDRecommender, ItemBasedCF
from .hybrid import HybridEnsemble

__all__ = [
    # Base
    "BaseRecommender",
    # Graph-based
    "LightGCN",
    "NGCF",
    # Neural
    "NCF",
    "GMF",
    "MLP",
    "NeuMF",
    # Traditional
    "SVDRecommender",
    "ItemBasedCF",
    # Hybrid
    "HybridEnsemble",
]
