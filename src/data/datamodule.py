"""
MovieLens DataModule for unified data loading and preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .graph_builder import GraphBuilder
from .negative_sampler import BPRDataset, NCFDataset, NegativeSampler
from .preprocessor import DataPreprocessor, LabelEncoderWrapper


@dataclass
class DataSplit:
    """Container for train/val/test data splits."""
    
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    
    # Interaction sets per split
    train_user_items: dict[int, set[int]]
    val_user_items: dict[int, set[int]]
    test_user_items: dict[int, set[int]]
    
    # Combined for negative sampling (exclude all positive interactions)
    all_user_items: dict[int, set[int]]


class MovieLensDataModule:
    """
    DataModule for MovieLens dataset.
    
    Handles:
    - Loading raw CSV files
    - Preprocessing and encoding
    - Train/val/test splitting
    - Graph construction for GNN models
    - Negative sampling for NCF/BPR
    - DataLoader creation
    """
    
    def __init__(
        self,
        data_dir: str | Path,
        cfg: Optional[DictConfig] = None,
        # Data files
        ratings_file: str = "ratings.csv",
        movies_file: str = "movies.csv",
        # Preprocessing
        min_rating: float = 3.0,
        min_user_interactions: int = 5,
        min_item_interactions: int = 5,
        # Split
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        split_strategy: str = "random",
        # DataLoader
        batch_size: int = 1024,
        num_workers: int = 4,
        pin_memory: bool = True,
        # Negative sampling
        num_negatives: int = 4,
        # Random seed
        seed: int = 42,
    ):
        """
        Initialize MovieLens DataModule.
        
        Args:
            data_dir: Directory containing MovieLens CSV files
            cfg: Optional Hydra config (overrides other args if provided)
            ratings_file: Name of ratings CSV file
            movies_file: Name of movies CSV file
            min_rating: Minimum rating to consider as positive interaction
            min_user_interactions: Filter users with fewer interactions
            min_item_interactions: Filter items with fewer interactions
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            split_strategy: Split strategy ('random', 'temporal', 'leave_one_out')
            batch_size: Batch size for DataLoaders
            num_workers: Number of workers for DataLoaders
            pin_memory: Whether to pin memory in DataLoaders
            num_negatives: Number of negative samples per positive
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        
        # Override with config if provided
        if cfg is not None:
            self.ratings_file = cfg.get("files", {}).get("ratings", ratings_file)
            self.movies_file = cfg.get("files", {}).get("movies", movies_file)
            self.min_rating = cfg.get("preprocessing", {}).get("min_rating", min_rating)
            self.min_user_interactions = cfg.get("preprocessing", {}).get(
                "min_user_interactions", min_user_interactions
            )
            self.min_item_interactions = cfg.get("preprocessing", {}).get(
                "min_item_interactions", min_item_interactions
            )
            self.train_ratio = cfg.get("split", {}).get("train_ratio", train_ratio)
            self.val_ratio = cfg.get("split", {}).get("val_ratio", val_ratio)
            self.test_ratio = cfg.get("split", {}).get("test_ratio", test_ratio)
            self.split_strategy = cfg.get("split", {}).get("strategy", split_strategy)
            self.batch_size = cfg.get("dataloader", {}).get("batch_size", batch_size)
            self.num_workers = cfg.get("dataloader", {}).get("num_workers", num_workers)
            self.pin_memory = cfg.get("dataloader", {}).get("pin_memory", pin_memory)
            self.num_negatives = cfg.get("negative_sampling", {}).get(
                "num_negatives", num_negatives
            )
            self.seed = seed
        else:
            self.ratings_file = ratings_file
            self.movies_file = movies_file
            self.min_rating = min_rating
            self.min_user_interactions = min_user_interactions
            self.min_item_interactions = min_item_interactions
            self.train_ratio = train_ratio
            self.val_ratio = val_ratio
            self.test_ratio = test_ratio
            self.split_strategy = split_strategy
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.pin_memory = pin_memory
            self.num_negatives = num_negatives
            self.seed = seed
        
        # Placeholders
        self.ratings_df: Optional[pd.DataFrame] = None
        self.movies_df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.data_split: Optional[DataSplit] = None
        self.graph_builder: Optional[GraphBuilder] = None
        
        # Cached tensors
        self._edge_index: Optional[torch.Tensor] = None
        self._edge_weights: Optional[torch.Tensor] = None
        
    def setup(self) -> None:
        """Load and preprocess data."""
        # Load raw data
        self._load_raw_data()
        
        # Preprocess
        self._preprocess()
        
        # Split
        self._split_data()
        
        # Build graph
        self._build_graph()
    
    def _load_raw_data(self) -> None:
        """Load raw CSV files."""
        ratings_path = self.data_dir / self.ratings_file
        movies_path = self.data_dir / self.movies_file
        
        if not ratings_path.exists():
            raise FileNotFoundError(f"Ratings file not found: {ratings_path}")
        
        self.ratings_df = pd.read_csv(ratings_path)
        
        if movies_path.exists():
            self.movies_df = pd.read_csv(movies_path)
    
    def _preprocess(self) -> None:
        """Preprocess ratings data."""
        self.preprocessor = DataPreprocessor(
            min_rating=self.min_rating,
            min_user_interactions=self.min_user_interactions,
            min_item_interactions=self.min_item_interactions,
        )
        
        self.processed_df = self.preprocessor.fit_transform(self.ratings_df)
    
    def _split_data(self) -> None:
        """Split data into train/val/test sets."""
        df = self.processed_df
        
        if self.split_strategy == "random":
            # Random split
            train_df, temp_df = train_test_split(
                df,
                test_size=(self.val_ratio + self.test_ratio),
                random_state=self.seed,
            )
            
            relative_test_ratio = self.test_ratio / (self.val_ratio + self.test_ratio)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=relative_test_ratio,
                random_state=self.seed,
            )
            
        elif self.split_strategy == "temporal":
            # Temporal split (requires timestamp)
            if "timestamp" not in df.columns:
                raise ValueError("Temporal split requires timestamp column")
            
            df_sorted = df.sort_values("timestamp")
            n = len(df_sorted)
            
            train_end = int(n * self.train_ratio)
            val_end = int(n * (self.train_ratio + self.val_ratio))
            
            train_df = df_sorted.iloc[:train_end]
            val_df = df_sorted.iloc[train_end:val_end]
            test_df = df_sorted.iloc[val_end:]
            
        elif self.split_strategy == "leave_one_out":
            # Leave-one-out: last interaction per user for test
            df_sorted = df.sort_values(["user_idx", "timestamp"])
            
            # Group by user and split
            train_list, val_list, test_list = [], [], []
            
            for _, group in df_sorted.groupby("user_idx"):
                if len(group) >= 3:
                    test_list.append(group.iloc[-1:])
                    val_list.append(group.iloc[-2:-1])
                    train_list.append(group.iloc[:-2])
                elif len(group) == 2:
                    test_list.append(group.iloc[-1:])
                    train_list.append(group.iloc[:-1])
                else:
                    train_list.append(group)
            
            train_df = pd.concat(train_list, ignore_index=True)
            val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
            test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()
        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")
        
        # Build user-item sets
        train_user_items, _ = self.preprocessor.get_user_item_sets(train_df)
        val_user_items, _ = self.preprocessor.get_user_item_sets(val_df) if len(val_df) > 0 else ({}, {})
        test_user_items, _ = self.preprocessor.get_user_item_sets(test_df) if len(test_df) > 0 else ({}, {})
        
        # Combined for negative sampling
        all_user_items, _ = self.preprocessor.get_user_item_sets(df)
        
        self.data_split = DataSplit(
            train_df=train_df.reset_index(drop=True),
            val_df=val_df.reset_index(drop=True),
            test_df=test_df.reset_index(drop=True),
            train_user_items=train_user_items,
            val_user_items=val_user_items,
            test_user_items=test_user_items,
            all_user_items=all_user_items,
        )
    
    def _build_graph(self) -> None:
        """Build graph for GNN models."""
        self.graph_builder = GraphBuilder(
            num_users=self.num_users,
            num_items=self.num_items,
            undirected=True,
            add_self_loops=False,
        )
        
        # Build from training data only
        train_df = self.data_split.train_df
        self._edge_index = self.graph_builder.build_edge_index(
            train_df["user_idx"].values,
            train_df["item_idx"].values,
        )
        self._edge_weights = self.graph_builder.compute_edge_weights(
            self._edge_index,
            normalization="sym",
        )
    
    @property
    def num_users(self) -> int:
        return self.preprocessor.num_users
    
    @property
    def num_items(self) -> int:
        return self.preprocessor.num_items
    
    @property
    def num_nodes(self) -> int:
        return self.num_users + self.num_items
    
    @property
    def edge_index(self) -> torch.Tensor:
        if self._edge_index is None:
            raise RuntimeError("Call setup() first")
        return self._edge_index
    
    @property
    def edge_weights(self) -> torch.Tensor:
        if self._edge_weights is None:
            raise RuntimeError("Call setup() first")
        return self._edge_weights
    
    @property
    def encoders(self) -> LabelEncoderWrapper:
        return self.preprocessor.encoders
    
    def get_bpr_dataloader(self, split: str = "train") -> DataLoader:
        """Get DataLoader for BPR training."""
        if split == "train":
            df = self.data_split.train_df
            user_item_set = self.data_split.all_user_items  # Exclude all positives
        elif split == "val":
            df = self.data_split.val_df
            user_item_set = self.data_split.all_user_items
        else:
            df = self.data_split.test_df
            user_item_set = self.data_split.all_user_items
        
        dataset = BPRDataset(
            user_indices=df["user_idx"].values,
            item_indices=df["item_idx"].values,
            num_items=self.num_items,
            user_item_set=user_item_set,
            num_negatives=1,  # BPR uses 1 negative per positive
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def get_ncf_dataloader(self, split: str = "train") -> DataLoader:
        """Get DataLoader for NCF training."""
        if split == "train":
            df = self.data_split.train_df
            user_item_set = self.data_split.all_user_items
            mode = "train"
        elif split == "val":
            df = self.data_split.val_df
            user_item_set = self.data_split.all_user_items
            mode = "eval"
        else:
            df = self.data_split.test_df
            user_item_set = self.data_split.all_user_items
            mode = "eval"
        
        # For eval, we need labels
        labels = np.ones(len(df), dtype=np.float32)
        
        dataset = NCFDataset(
            user_indices=df["user_idx"].values,
            item_indices=df["item_idx"].values,
            labels=labels,
            num_items=self.num_items,
            user_item_set=user_item_set,
            num_negatives=self.num_negatives,
            mode=mode,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def get_evaluation_data(self, split: str = "test") -> tuple[dict, dict]:
        """
        Get evaluation data for ranking metrics.
        
        Returns:
            (ground_truth, train_items)
            - ground_truth: dict mapping user_idx -> set of positive item_idx in split
            - train_items: dict mapping user_idx -> set of positive item_idx in training
        """
        if split == "val":
            ground_truth = self.data_split.val_user_items
        else:
            ground_truth = self.data_split.test_user_items
        
        train_items = self.data_split.train_user_items
        
        return ground_truth, train_items
    
    def get_movie_info(self, item_indices: list[int]) -> pd.DataFrame:
        """Get movie information for item indices."""
        if self.movies_df is None:
            return pd.DataFrame()
        
        movie_ids = self.encoders.inverse_transform_items(np.array(item_indices))
        return self.movies_df[self.movies_df["movieId"].isin(movie_ids)]
