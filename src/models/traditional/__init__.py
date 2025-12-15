"""
Traditional recommendation methods.
"""
from .svd import SVDRecommender
from .item_cf import ItemBasedCF

__all__ = [
    "SVDRecommender",
    "ItemBasedCF",
]
