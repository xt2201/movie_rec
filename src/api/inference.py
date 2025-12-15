"""
Inference service for loading models and generating recommendations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.base import BaseRecommender
from src.models.graph import LightGCN, NGCF
from src.models.neural import NCF
from src.retrieval import FAISSRetriever


@dataclass
class ModelRegistry:
    """Registry for managing multiple recommendation models."""
    
    models: dict[str, BaseRecommender] = field(default_factory=dict)
    default_model: str | None = None
    
    def register(self, name: str, model: BaseRecommender, set_default: bool = False) -> None:
        """Register a model."""
        self.models[name] = model
        if set_default or self.default_model is None:
            self.default_model = name
    
    def get(self, name: str | None = None) -> BaseRecommender:
        """Get a model by name or default."""
        name = name or self.default_model
        if name is None:
            raise ValueError("No model registered")
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found. Available: {list(self.models.keys())}")
        return self.models[name]
    
    def list_models(self) -> list[str]:
        """List all registered models."""
        return list(self.models.keys())


class InferenceService:
    """
    Service for loading models and generating recommendations.
    
    Handles:
    - Model loading from checkpoints or MLflow
    - User/item encoding
    - Recommendation generation
    - Caching and batch inference
    """
    
    def __init__(
        self,
        model_dir: str | Path,
        data_dir: str | Path,
        device: str = "cpu",
    ):
        """
        Initialize inference service.
        
        Args:
            model_dir: Directory containing model checkpoints
            data_dir: Directory containing data files and encoders
            device: Device to run inference on
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.device = torch.device(device)
        
        self.registry = ModelRegistry()
        self.user_encoder: dict[int, int] = {}
        self.item_encoder: dict[int, int] = {}
        self.item_decoder: dict[int, int] = {}
        
        self.movie_details: dict[int, dict] = {}
        self.faiss_retriever: FAISSRetriever | None = None
        
        # Graph data for GNN models
        self.edge_index: torch.Tensor | None = None
        self.edge_weight: torch.Tensor | None = None
        
        self._initialized = False
    
    def initialize(self) -> None:
        """Load encoders, movie details, and models."""
        self._load_encoders()
        self._load_movie_details()
        self._load_graph_data()
        self._initialized = True
    
    def _load_encoders(self) -> None:
        """Load user and item encoders."""
        encoder_path = self.data_dir / "encoders.json"
        if encoder_path.exists():
            with open(encoder_path) as f:
                data = json.load(f)
            self.user_encoder = {int(k): v for k, v in data.get("user_encoder", {}).items()}
            self.item_encoder = {int(k): v for k, v in data.get("item_encoder", {}).items()}
            self.item_decoder = {v: k for k, v in self.item_encoder.items()}
        else:
            # Try loading from preprocessor pickle
            preprocessor_path = self.data_dir / "preprocessor.pkl"
            if preprocessor_path.exists():
                import pickle
                with open(preprocessor_path, "rb") as f:
                    preprocessor = pickle.load(f)
                self.user_encoder = dict(zip(
                    preprocessor.user_encoder.classes_,
                    range(len(preprocessor.user_encoder.classes_))
                ))
                self.item_encoder = dict(zip(
                    preprocessor.item_encoder.classes_,
                    range(len(preprocessor.item_encoder.classes_))
                ))
                self.item_decoder = {v: k for k, v in self.item_encoder.items()}
    
    def _load_movie_details(self) -> None:
        """Load movie metadata."""
        # Try different possible locations
        possible_paths = [
            self.data_dir / "movie_details.json",
            self.data_dir / "unique_movie_details.json",
            self.data_dir.parent / "unique_movie_details.json",
        ]
        
        for path in possible_paths:
            if path.exists():
                with open(path) as f:
                    self.movie_details = json.load(f)
                # Convert keys to int if they're strings
                self.movie_details = {
                    int(k) if isinstance(k, str) else k: v 
                    for k, v in self.movie_details.items()
                }
                break
        
        # If no details file, try loading from movies.csv
        if not self.movie_details:
            movies_path = self.data_dir / "movies.csv"
            if movies_path.exists():
                import pandas as pd
                movies_df = pd.read_csv(movies_path)
                for _, row in movies_df.iterrows():
                    self.movie_details[row["movieId"]] = {
                        "title": row["title"],
                        "genres": row.get("genres", "").split("|") if row.get("genres") else [],
                    }
    
    def _load_graph_data(self) -> None:
        """Load precomputed graph data for GNN models."""
        edge_index_path = self.data_dir / "edge_index.pt"
        if edge_index_path.exists():
            self.edge_index = torch.load(edge_index_path, map_location=self.device)
        
        edge_weight_path = self.data_dir / "edge_weights.pt"
        if edge_weight_path.exists():
            self.edge_weight = torch.load(edge_weight_path, map_location=self.device)
    
    def load_model(
        self,
        name: str,
        model_type: str,
        checkpoint_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
        set_default: bool = False,
    ) -> None:
        """
        Load a model from checkpoint.
        
        Args:
            name: Name to register the model under
            model_type: Type of model (lightgcn, ngcf, ncf)
            checkpoint_path: Path to model checkpoint
            config: Model configuration
            set_default: Whether to set as default model
        """
        if not self._initialized:
            self.initialize()
        
        num_users = len(self.user_encoder)
        num_items = len(self.item_encoder)
        config = config or {}
        
        # Build model
        model_type = model_type.lower()
        if model_type == "lightgcn":
            model = LightGCN(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=config.get("embedding_dim", 64),
                num_layers=config.get("num_layers", 3),
                dropout=config.get("dropout", 0.0),
                device=str(self.device),
            )
        elif model_type == "ngcf":
            model = NGCF(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=config.get("embedding_dim", 64),
                hidden_dims=config.get("hidden_dims", [64, 64]),
                dropout=config.get("dropout", 0.1),
                device=str(self.device),
            )
        elif model_type == "ncf":
            model = NCF(
                num_users=num_users,
                num_items=num_items,
                mf_dim=config.get("mf_dim", 32),
                mlp_layers=config.get("mlp_layers", [128, 64, 32]),
                dropout=config.get("dropout", 0.0),
                mode=config.get("mode", "neumf"),
                device=str(self.device),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights if provided
        if checkpoint_path:
            model.load(Path(checkpoint_path))
        
        model.to(self.device)
        model.eval()
        
        self.registry.register(name, model, set_default=set_default)
    
    def load_from_mlflow(
        self,
        name: str,
        model_uri: str,
        set_default: bool = False,
    ) -> None:
        """
        Load a model from MLflow model registry.
        
        Args:
            name: Name to register the model under
            model_uri: MLflow model URI (e.g., "models:/model_name/1")
            set_default: Whether to set as default model
        """
        try:
            import mlflow.pytorch
            model = mlflow.pytorch.load_model(model_uri)
            model.to(self.device)
            model.eval()
            self.registry.register(name, model, set_default=set_default)
        except ImportError:
            raise ImportError("MLflow is required for loading from model registry")
    
    def build_faiss_index(
        self,
        model_name: str | None = None,
        index_type: str = "flat",
        **kwargs,
    ) -> None:
        """
        Build FAISS index from model embeddings.
        
        Args:
            model_name: Name of model to get embeddings from
            index_type: Type of FAISS index
            **kwargs: Additional arguments for FAISS index
        """
        model = self.registry.get(model_name)
        
        # Get item embeddings
        with torch.no_grad():
            if hasattr(model, "get_embeddings"):
                if self.edge_index is not None:
                    _, item_embeddings = model.get_embeddings(self.edge_index, self.edge_weight)
                else:
                    _, item_embeddings = model.get_embeddings()
            else:
                # For NCF, get item embeddings from MF component
                item_ids = torch.arange(len(self.item_encoder), device=self.device)
                item_embeddings = model.gmf_item_embedding(item_ids)
        
        item_embeddings = item_embeddings.cpu().numpy()
        
        self.faiss_retriever = FAISSRetriever(
            dimension=item_embeddings.shape[1],
            index_type=index_type,
            **kwargs,
        )
        self.faiss_retriever.build_index(item_embeddings)
    
    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        model_name: str | None = None,
        exclude_interacted: bool = True,
        interacted_items: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: Original user ID (before encoding)
            top_k: Number of recommendations
            model_name: Name of model to use
            exclude_interacted: Whether to exclude already interacted items
            interacted_items: List of already interacted item IDs (original IDs)
            
        Returns:
            List of recommendation dictionaries with movie details
        """
        if not self._initialized:
            self.initialize()
        
        model = self.registry.get(model_name)
        
        # Encode user
        if user_id not in self.user_encoder:
            raise ValueError(f"Unknown user ID: {user_id}")
        
        encoded_user = self.user_encoder[user_id]
        user_tensor = torch.tensor([encoded_user], device=self.device)
        
        # Get exclude set
        exclude_set = set()
        if exclude_interacted and interacted_items:
            exclude_set = {
                self.item_encoder.get(item_id) 
                for item_id in interacted_items 
                if item_id in self.item_encoder
            }
            exclude_set.discard(None)
        
        # Get recommendations
        with torch.no_grad():
            if hasattr(model, "recommend"):
                # Graph models
                recommended = model.recommend(
                    user_tensor,
                    top_k=top_k + len(exclude_set),  # Extra in case of exclusions
                    edge_index=self.edge_index,
                    edge_weight=self.edge_weight,
                )
            else:
                # NCF models - score all items
                all_items = torch.arange(len(self.item_encoder), device=self.device)
                users = user_tensor.expand(len(all_items))
                scores = model.predict(users, all_items)
                _, indices = torch.topk(scores, top_k + len(exclude_set))
                recommended = indices.unsqueeze(0)
        
        # Convert to list and filter
        recommendations = []
        for item_idx in recommended[0].cpu().numpy():
            if int(item_idx) in exclude_set:
                continue
            
            original_item_id = self.item_decoder.get(int(item_idx))
            if original_item_id is None:
                continue
            
            rec = {
                "item_id": original_item_id,
                "encoded_id": int(item_idx),
            }
            
            # Add movie details
            if original_item_id in self.movie_details:
                rec.update(self.movie_details[original_item_id])
            
            recommendations.append(rec)
            
            if len(recommendations) >= top_k:
                break
        
        return recommendations
    
    def recommend_similar(
        self,
        item_id: int,
        top_k: int = 10,
        model_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get similar items based on model embeddings.
        
        Args:
            item_id: Original item ID
            top_k: Number of similar items
            model_name: Name of model to use
            
        Returns:
            List of similar item dictionaries
        """
        if not self._initialized:
            self.initialize()
        
        if self.faiss_retriever is None:
            self.build_faiss_index(model_name)
        
        # Encode item
        if item_id not in self.item_encoder:
            raise ValueError(f"Unknown item ID: {item_id}")
        
        encoded_item = self.item_encoder[item_id]
        
        # Get item embedding
        model = self.registry.get(model_name)
        with torch.no_grad():
            if hasattr(model, "get_embeddings"):
                if self.edge_index is not None:
                    _, item_embeddings = model.get_embeddings(self.edge_index, self.edge_weight)
                else:
                    _, item_embeddings = model.get_embeddings()
                query = item_embeddings[encoded_item:encoded_item+1].cpu().numpy()
            else:
                item_tensor = torch.tensor([encoded_item], device=self.device)
                query = model.gmf_item_embedding(item_tensor).cpu().numpy()
        
        # Search
        indices, distances = self.faiss_retriever.search(query, top_k + 1)
        
        # Convert to list
        similar_items = []
        for idx, dist in zip(indices[0], distances[0]):
            if int(idx) == encoded_item:  # Skip the query item
                continue
            
            original_id = self.item_decoder.get(int(idx))
            if original_id is None:
                continue
            
            item = {
                "item_id": original_id,
                "encoded_id": int(idx),
                "similarity": float(1.0 / (1.0 + dist)),
            }
            
            if original_id in self.movie_details:
                item.update(self.movie_details[original_id])
            
            similar_items.append(item)
            
            if len(similar_items) >= top_k:
                break
        
        return similar_items
    
    def batch_recommend(
        self,
        user_ids: list[int],
        top_k: int = 10,
        model_name: str | None = None,
    ) -> dict[int, list[dict[str, Any]]]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids: List of original user IDs
            top_k: Number of recommendations per user
            model_name: Name of model to use
            
        Returns:
            Dictionary mapping user IDs to recommendations
        """
        results = {}
        for user_id in user_ids:
            try:
                results[user_id] = self.recommend(user_id, top_k, model_name)
            except ValueError:
                results[user_id] = []
        return results
