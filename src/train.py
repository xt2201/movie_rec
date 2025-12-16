"""
Main training script with Hydra configuration.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.data import MovieLensDataModule
from src.evaluation import Evaluator
from src.models.graph import LightGCN, NGCF
from src.models.neural import NCF
from src.training import Trainer, TrainerConfig
from src.utils import ExperimentTracker
from src.utils.rich_logging import console, display_config


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_cfg: str) -> torch.device:
    """Get device from config."""
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_cfg)


def build_model(cfg: DictConfig, num_users: int, num_items: int, device: torch.device):
    """Build model based on config."""
    model_name = cfg.model.name.lower()
    
    if model_name == "lightgcn":
        model = LightGCN(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=cfg.model.embedding_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            device=str(device),
        )
    elif model_name == "ngcf":
        model = NGCF(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=cfg.model.embedding_dim,
            hidden_dims=cfg.model.hidden_dims,
            dropout=cfg.model.dropout,
            device=str(device),
        )
    elif model_name == "ncf":
        model = NCF(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=cfg.model.get("embedding_dim", 64),
            mf_dim=cfg.model.mf_dim,
            mlp_layers=cfg.model.mlp_layers,
            dropout=cfg.model.dropout,
            mode="neumf",
            device=str(device),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Main training function.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Best validation loss for hyperparameter optimization
    """
    # Print config
    console.rule("[bold]Movie Recommendation Training[/bold]")
    display_config(OmegaConf.to_container(cfg, resolve=True))
    
    # Set seed
    set_seed(cfg.seed)
    
    # Get device
    device = get_device(cfg.device)
    console.print(f"[cyan]Using device:[/cyan] {device}")
    
    # Setup data
    console.rule("[bold]Loading Data[/bold]")
    
    data_module = MovieLensDataModule(
        data_dir=cfg.paths.data_dir,
        cfg=cfg.data,
        seed=cfg.seed,
    )
    data_module.setup()
    
    console.print(f"[green]✓[/green] Loaded {data_module.num_users:,} users, {data_module.num_items:,} items")
    console.print(f"[green]✓[/green] Train: {len(data_module.data_split.train_df):,} interactions")
    console.print(f"[green]✓[/green] Val: {len(data_module.data_split.val_df):,} interactions")
    console.print(f"[green]✓[/green] Test: {len(data_module.data_split.test_df):,} interactions")
    
    # Build model
    console.rule("[bold]Building Model[/bold]")
    
    model = build_model(cfg, data_module.num_users, data_module.num_items, device)
    console.print(f"[green]✓[/green] Built {model}")
    
    # Setup experiment tracking
    tracker = ExperimentTracker(
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        config=cfg,
        use_mlflow=cfg.logger.use_mlflow,
        mlflow_tracking_uri=cfg.paths.mlflow_dir,
        use_wandb=cfg.logger.use_wandb,
        wandb_project=cfg.logger.wandb.project,
        wandb_entity=cfg.logger.wandb.get("entity"),
        wandb_tags=cfg.logger.wandb.get("tags", []),
        watch_model=cfg.logger.wandb.get("watch_model", False),
    )
    
    # Setup trainer
    trainer_config = TrainerConfig(
        max_epochs=cfg.trainer.max_epochs,
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        batch_size=cfg.trainer.batch_size,
        optimizer=cfg.trainer.optimizer.name,
        scheduler=cfg.trainer.scheduler.name,
        early_stopping=cfg.trainer.early_stopping.enabled,
        patience=cfg.trainer.early_stopping.patience,
        gradient_clip=cfg.trainer.gradient_clip.enabled,
        max_grad_norm=cfg.trainer.gradient_clip.max_norm,
        checkpoint_dir=cfg.paths.checkpoint_dir,
        reg_weight=cfg.trainer.get("reg_weight", 1e-5),
        eval_every_n_epochs=cfg.trainer.get("eval_every_n_epochs", 5),
        seed=cfg.seed,
    )
    
    # Setup evaluator for validation metrics
    val_evaluator = Evaluator(
        k_values=[5, 10, 20],
        metrics=["precision", "recall", "ndcg", "hit_rate"],
    )
    
    # Get validation ground truth
    val_ground_truth, val_train_items = data_module.get_evaluation_data("val")
    
    trainer = Trainer(
        model=model,
        config=trainer_config,
        tracker=tracker,
        device=str(device),
        evaluator=val_evaluator,
        val_ground_truth=val_ground_truth,
        val_train_items=val_train_items,
        num_items=data_module.num_items,
    )
    
    # Get data loaders
    model_type = cfg.model.get("type", "graph")
    
    # All models use BPR-style dataloader (user, pos_item, neg_item)
    train_loader = data_module.get_bpr_dataloader("train")
    val_loader = data_module.get_bpr_dataloader("val")
    
    if model_type == "graph":
        edge_index = data_module.edge_index.to(device)
        # Don't pass precomputed edge_weight - let LightGCNConv compute normalization itself
        # This avoids double-normalization bug
        edge_weight = None
    else:
        edge_index = None
        edge_weight = None
    
    # Train
    console.rule("[bold]Training[/bold]")
    
    with tracker:
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
    
    # Evaluate
    console.rule("[bold]Evaluation[/bold]")
    
    evaluator = Evaluator(
        k_values=[5, 10, 20],
        metrics=["precision", "recall", "ndcg", "hit_rate"],
    )
    
    ground_truth, train_items = data_module.get_evaluation_data("test")
    
    results = evaluator.evaluate(
        model=model,
        ground_truth=ground_truth,
        train_items=train_items,
        num_items=data_module.num_items,
        edge_index=edge_index,
        edge_weight=edge_weight,
        device=device,
    )
    
    evaluator.print_results(results)
    
    # Log final metrics
    if tracker:
        for metric, value in results.items():
            tracker.set_summary(metric, value)
    
    # Save final model
    model_path = Path(cfg.paths.checkpoint_dir) / "final_model.pt"
    model.save(model_path)
    console.print(f"[green]✓[/green] Saved model to {model_path}")
    
    # Save results as JSON for benchmark integration
    import json
    results_path = Path(cfg.paths.checkpoint_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    console.rule("[bold green]Training Complete![/bold green]")
    
    return min(history.get("val_loss", [float("inf")]))


if __name__ == "__main__":
    main()
