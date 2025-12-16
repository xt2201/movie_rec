"""
Unified evaluator for recommendation models.
"""
from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.rich_logging import console, RichLogger
from .metrics import (
    batch_hit_rate_at_k,
    batch_ndcg_at_k,
    batch_precision_at_k,
    batch_recall_at_k,
    coverage,
)


class Evaluator:
    """
    Unified evaluator for recommendation models.
    
    Supports:
    - Multiple K values
    - Multiple metrics
    - Batch evaluation
    - Coverage computation
    """
    
    SUPPORTED_METRICS = [
        "precision",
        "recall",
        "ndcg",
        "hit_rate",
        "map",
        "mrr",
        "coverage",
    ]
    
    def __init__(
        self,
        k_values: list[int] = [5, 10, 20],
        metrics: Optional[list[str]] = None,
        batch_size: int = 256,
    ):
        """
        Initialize evaluator.
        
        Args:
            k_values: List of K values for @K metrics
            metrics: Metrics to compute (default: all)
            batch_size: Batch size for evaluation
        """
        self.k_values = k_values
        self.metrics = metrics or ["precision", "recall", "ndcg", "hit_rate"]
        self.batch_size = batch_size
        
        self.logger = RichLogger()
    
    def evaluate(
        self,
        model,
        ground_truth: dict[int, set[int]],
        train_items: dict[int, set[int]],
        num_items: int,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        user_emb: Optional[torch.Tensor] = None,
        item_emb: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            model: Recommendation model
            ground_truth: Dict mapping user_idx -> set of test positive items
            train_items: Dict mapping user_idx -> set of training items (to exclude)
            num_items: Total number of items
            edge_index: Graph edge index (for GNN models)
            edge_weight: Edge weights
            device: Device to use
            user_emb: Precomputed user embeddings (to avoid recomputation)
            item_emb: Precomputed item embeddings (to avoid recomputation)
            
        Returns:
            Dictionary of metric_name -> value
        """
        model.eval()
        
        if device is None:
            device = next(model.parameters()).device
        
        # Get all test users
        test_users = sorted(ground_truth.keys())
        
        if not test_users:
            return {}
        
        max_k = max(self.k_values)
        
        # Use precomputed embeddings if provided, otherwise compute them
        if user_emb is None or item_emb is None:
            if hasattr(model, "forward") and edge_index is not None:
                with torch.no_grad():
                    if edge_index is not None:
                        edge_index = edge_index.to(device)
                    if edge_weight is not None:
                        edge_weight = edge_weight.to(device)
                    
                    user_emb, item_emb = model.forward(edge_index, edge_weight)
        
        # Evaluate in batches
        all_predictions = []
        
        console.print("[cyan]Evaluating model...[/cyan]")
        
        for i in tqdm(range(0, len(test_users), self.batch_size), desc="Evaluation"):
            batch_users = test_users[i:i + self.batch_size]
            user_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
            
            # Get recommendations
            with torch.no_grad():
                if user_emb is not None:
                    _, top_items = model.recommend(
                        user_tensor,
                        k=max_k,
                        exclude_items=train_items,
                        user_emb=user_emb,
                        item_emb=item_emb,
                    )
                else:
                    _, top_items = model.recommend(
                        user_tensor,
                        k=max_k,
                        exclude_items=train_items,
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                    )
            
            all_predictions.append(top_items.cpu().numpy())
        
        # Concatenate predictions
        predictions = np.concatenate(all_predictions, axis=0)
        
        # Build ground truth dict with sequential indices
        gt_dict = {i: ground_truth[user] for i, user in enumerate(test_users)}
        
        # Compute metrics (use _at_ instead of @ for MLflow compatibility)
        results = {}
        
        for k in self.k_values:
            preds_k = predictions[:, :k]
            
            if "precision" in self.metrics:
                results[f"precision_at_{k}"] = batch_precision_at_k(preds_k, gt_dict, k)
            
            if "recall" in self.metrics:
                results[f"recall_at_{k}"] = batch_recall_at_k(preds_k, gt_dict, k)
            
            if "ndcg" in self.metrics:
                results[f"ndcg_at_{k}"] = batch_ndcg_at_k(preds_k, gt_dict, k)
            
            if "hit_rate" in self.metrics:
                results[f"hit_rate_at_{k}"] = batch_hit_rate_at_k(preds_k, gt_dict, k)
        
        # Coverage
        if "coverage" in self.metrics:
            results["coverage"] = coverage(
                [predictions[i] for i in range(len(predictions))],
                num_items,
                max_k,
            )
        
        return results
    
    def print_results(self, results: dict[str, float]) -> None:
        """Print evaluation results in a formatted table."""
        from rich.table import Table
        
        table = Table(title="Evaluation Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        
        for metric, value in sorted(results.items()):
            table.add_row(metric, f"{value:.4f}")
        
        console.print(table)
    
    def compare_models(
        self,
        results_dict: dict[str, dict[str, float]],
    ) -> None:
        """
        Compare results from multiple models.
        
        Args:
            results_dict: Dict mapping model_name -> results
        """
        from rich.table import Table
        
        # Get all metrics
        all_metrics = set()
        for results in results_dict.values():
            all_metrics.update(results.keys())
        all_metrics = sorted(all_metrics)
        
        # Create table
        table = Table(title="Model Comparison", show_header=True)
        table.add_column("Metric", style="cyan")
        
        for model_name in results_dict.keys():
            table.add_column(model_name, justify="right")
        
        # Add rows
        for metric in all_metrics:
            row = [metric]
            values = []
            
            for model_name, results in results_dict.items():
                value = results.get(metric, 0)
                values.append(value)
                row.append(f"{value:.4f}")
            
            # Highlight best value
            best_idx = np.argmax(values)
            row[best_idx + 1] = f"[bold green]{row[best_idx + 1]}[/bold green]"
            
            table.add_row(*row)
        
        console.print(table)
