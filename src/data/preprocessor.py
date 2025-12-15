"""
Data preprocessing utilities for recommendation systems.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@dataclass
class LabelEncoderWrapper:
    """Wrapper for user/item label encoders with save/load support."""
    
    user_encoder: LabelEncoder = field(default_factory=LabelEncoder)
    item_encoder: LabelEncoder = field(default_factory=LabelEncoder)
    _fitted: bool = False
    
    @property
    def num_users(self) -> int:
        if not self._fitted:
            raise RuntimeError("Encoders not fitted. Call fit() first.")
        return len(self.user_encoder.classes_)
    
    @property
    def num_items(self) -> int:
        if not self._fitted:
            raise RuntimeError("Encoders not fitted. Call fit() first.")
        return len(self.item_encoder.classes_)
    
    def fit(self, user_ids: np.ndarray, item_ids: np.ndarray) -> "LabelEncoderWrapper":
        """Fit encoders on user and item IDs."""
        self.user_encoder.fit(user_ids)
        self.item_encoder.fit(item_ids)
        self._fitted = True
        return self
    
    def transform_users(self, user_ids: np.ndarray) -> np.ndarray:
        """Transform user IDs to encoded indices."""
        return self.user_encoder.transform(user_ids)
    
    def transform_items(self, item_ids: np.ndarray) -> np.ndarray:
        """Transform item IDs to encoded indices."""
        return self.item_encoder.transform(item_ids)
    
    def inverse_transform_users(self, encoded_ids: np.ndarray) -> np.ndarray:
        """Transform encoded indices back to original user IDs."""
        return self.user_encoder.inverse_transform(encoded_ids)
    
    def inverse_transform_items(self, encoded_ids: np.ndarray) -> np.ndarray:
        """Transform encoded indices back to original item IDs."""
        return self.item_encoder.inverse_transform(encoded_ids)
    
    def save(self, path: Path) -> None:
        """Save encoders to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "user_encoder": self.user_encoder,
                "item_encoder": self.item_encoder,
            }, f)
    
    @classmethod
    def load(cls, path: Path) -> "LabelEncoderWrapper":
        """Load encoders from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        wrapper = cls()
        wrapper.user_encoder = data["user_encoder"]
        wrapper.item_encoder = data["item_encoder"]
        wrapper._fitted = True
        return wrapper


class DataPreprocessor:
    """Preprocessor for recommendation datasets."""
    
    def __init__(
        self,
        min_rating: float = 3.0,
        min_user_interactions: int = 5,
        min_item_interactions: int = 5,
    ):
        self.min_rating = min_rating
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        self.encoders = LabelEncoderWrapper()
        
    def fit_transform(
        self,
        ratings_df: pd.DataFrame,
        user_col: str = "userId",
        item_col: str = "movieId", 
        rating_col: str = "rating",
        timestamp_col: Optional[str] = "timestamp",
    ) -> pd.DataFrame:
        """
        Preprocess ratings dataframe.
        
        Steps:
        1. Filter by minimum rating (implicit feedback)
        2. Filter users with too few interactions
        3. Filter items with too few interactions
        4. Encode user/item IDs to contiguous integers
        
        Args:
            ratings_df: Raw ratings dataframe
            user_col: Name of user ID column
            item_col: Name of item ID column
            rating_col: Name of rating column
            timestamp_col: Name of timestamp column (optional)
            
        Returns:
            Preprocessed dataframe with encoded IDs
        """
        df = ratings_df.copy()
        
        # Step 1: Filter by minimum rating for implicit feedback
        if self.min_rating > 0:
            df = df[df[rating_col] >= self.min_rating]
        
        # Step 2 & 3: Iteratively filter users and items
        # (may need multiple passes as filtering affects counts)
        prev_len = 0
        while len(df) != prev_len:
            prev_len = len(df)
            
            # Filter users
            user_counts = df[user_col].value_counts()
            valid_users = user_counts[user_counts >= self.min_user_interactions].index
            df = df[df[user_col].isin(valid_users)]
            
            # Filter items  
            item_counts = df[item_col].value_counts()
            valid_items = item_counts[item_counts >= self.min_item_interactions].index
            df = df[df[item_col].isin(valid_items)]
        
        # Step 4: Encode IDs
        self.encoders.fit(df[user_col].values, df[item_col].values)
        
        df["user_idx"] = self.encoders.transform_users(df[user_col].values)
        df["item_idx"] = self.encoders.transform_items(df[item_col].values)
        
        # Keep timestamp if exists
        columns_to_keep = ["user_idx", "item_idx", user_col, item_col, rating_col]
        if timestamp_col and timestamp_col in df.columns:
            columns_to_keep.append(timestamp_col)
            
        return df[columns_to_keep].reset_index(drop=True)
    
    def get_interaction_matrix(
        self,
        df: pd.DataFrame,
        sparse: bool = True,
    ) -> np.ndarray:
        """
        Create user-item interaction matrix.
        
        Args:
            df: Preprocessed dataframe with user_idx, item_idx columns
            sparse: Whether to return scipy sparse matrix
            
        Returns:
            Interaction matrix of shape (num_users, num_items)
        """
        from scipy.sparse import csr_matrix
        
        num_users = self.encoders.num_users
        num_items = self.encoders.num_items
        
        rows = df["user_idx"].values
        cols = df["item_idx"].values
        data = np.ones(len(df))
        
        matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(num_users, num_items),
            dtype=np.float32,
        )
        
        if sparse:
            return matrix
        return matrix.toarray()
    
    def get_user_item_sets(
        self,
        df: pd.DataFrame,
    ) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
        """
        Get dictionaries mapping users to their items and vice versa.
        
        Returns:
            (user_to_items, item_to_users) dictionaries
        """
        user_to_items: dict[int, set[int]] = {}
        item_to_users: dict[int, set[int]] = {}
        
        for _, row in df.iterrows():
            user_idx = int(row["user_idx"])
            item_idx = int(row["item_idx"])
            
            if user_idx not in user_to_items:
                user_to_items[user_idx] = set()
            user_to_items[user_idx].add(item_idx)
            
            if item_idx not in item_to_users:
                item_to_users[item_idx] = set()
            item_to_users[item_idx].add(user_idx)
        
        return user_to_items, item_to_users
    
    @property
    def num_users(self) -> int:
        return self.encoders.num_users
    
    @property
    def num_items(self) -> int:
        return self.encoders.num_items
