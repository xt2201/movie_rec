"""
Generic trainer for recommendation models.

Supports:
- GNN models (LightGCN, NGCF)
- Neural models (NCF)
- Multiple loss functions (BPR, BCE)
- Callbacks (early stopping, checkpointing)
- Experiment tracking (MLflow, W&B)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from ..evaluation import Evaluator
from ..utils.experiment_tracker import ExperimentTracker
from ..utils.rich_logging import (
    RichLogger,
    console,
    create_progress,
    display_metrics_table,
    display_model_summary,
)
from .callbacks import Callback, EarlyStopping, ModelCheckpoint


@dataclass
class TrainerConfig:
    """Trainer configuration."""
    
    # Training
    max_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 1024
    
    # Optimizer
    optimizer: str = "adam"
    betas: tuple[float, float] = (0.9, 0.999)
    
    # Scheduler
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.0001
    monitor: str = "val_loss"
    
    # Gradient clipping
    gradient_clip: bool = True
    max_grad_norm: float = 1.0
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 3
    save_last: bool = True
    
    # Validation
    val_check_interval: int = 1
    eval_every_n_epochs: int = 5  # Run full ranking evaluation every N epochs
    
    # Regularization
    reg_weight: float = 1e-5
    
    # Misc
    seed: int = 42
    detect_anomaly: bool = False


class Trainer:
    """
    Generic trainer for recommendation models.
    
    Handles:
    - Training loop with progress bars
    - Validation
    - Callbacks (early stopping, checkpointing)
    - Experiment tracking
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        tracker: Optional[ExperimentTracker] = None,
        callbacks: Optional[list[Callback]] = None,
        device: Optional[str] = None,
        evaluator: Optional[Evaluator] = None,
        val_ground_truth: Optional[dict[int, set[int]]] = None,
        val_train_items: Optional[dict[int, set[int]]] = None,
        num_items: Optional[int] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Trainer configuration
            tracker: Experiment tracker
            callbacks: List of callbacks
            device: Device to use
            evaluator: Evaluator for ranking metrics during validation
            val_ground_truth: Ground truth for validation (user_idx -> positive item_idx)
            val_train_items: Training items to exclude during validation
            num_items: Total number of items (for evaluation)
        """
        self.model = model
        self.config = config
        self.tracker = tracker
        self.callbacks = callbacks or []
        
        # Evaluation components
        self.evaluator = evaluator
        self.val_ground_truth = val_ground_truth
        self.val_train_items = val_train_items
        self.num_items = num_items
        
        # Set device
        if device is None or device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Add default callbacks
        self._setup_default_callbacks()
        
        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        
        # Logger
        self.logger = RichLogger()
        
        # Anomaly detection
        if config.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
    
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer based on config."""
        params = self.model.parameters()
        
        if self.config.optimizer == "adam":
            return Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
            )
        elif self.config.optimizer == "adamw":
            return AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
            )
        elif self.config.optimizer == "sgd":
            return SGD(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler == "reduce_on_plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                min_lr=self.config.min_lr,
            )
        elif self.config.scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler == "step":
            return StepLR(
                self.optimizer,
                step_size=self.config.scheduler_patience,
                gamma=self.config.scheduler_factor,
            )
        elif self.config.scheduler == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def _setup_default_callbacks(self) -> None:
        """Setup default callbacks if not provided."""
        # Check if early stopping already exists
        has_early_stopping = any(
            isinstance(cb, EarlyStopping) for cb in self.callbacks
        )
        
        if self.config.early_stopping and not has_early_stopping:
            self.callbacks.append(
                EarlyStopping(
                    monitor=self.config.monitor,
                    patience=self.config.patience,
                    min_delta=self.config.min_delta,
                )
            )
        
        # Check if checkpoint already exists
        has_checkpoint = any(
            isinstance(cb, ModelCheckpoint) for cb in self.callbacks
        )
        
        if not has_checkpoint:
            self.callbacks.append(
                ModelCheckpoint(
                    dirpath=self.config.checkpoint_dir,
                    monitor=self.config.monitor,
                    save_top_k=self.config.save_top_k,
                    save_last=self.config.save_last,
                )
            )
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> dict[str, list[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            edge_index: Graph edge index (for GNN models)
            edge_weight: Edge weights (for GNN models)
            
        Returns:
            Training history
        """
        # Move graph data to device
        if edge_index is not None:
            edge_index = edge_index.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)
        
        # Display model summary
        display_model_summary(self.model)
        
        # Callback: train start
        for cb in self.callbacks:
            cb.on_train_start(self)
        
        # Watch model gradients
        if self.tracker and hasattr(self.tracker, "watch_model_gradients"):
            self.tracker.watch_model_gradients(self.model)
        
        # Training history
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }
        
        # Training loop with progress
        with create_progress() as progress:
            epoch_task = progress.add_task(
                "[cyan]Training", total=self.config.max_epochs
            )
            
            for epoch in range(self.config.max_epochs):
                self.current_epoch = epoch
                
                # Callback: epoch start
                for cb in self.callbacks:
                    cb.on_epoch_start(self, epoch)
                
                # Training
                train_metrics = self._train_epoch(
                    train_loader,
                    edge_index,
                    edge_weight,
                    progress,
                )
                history["train_loss"].append(train_metrics["loss"])
                
                # Validation
                val_metrics = {}
                if val_loader is not None and epoch % self.config.val_check_interval == 0:
                    # Check if we should run full ranking evaluation this epoch
                    run_eval = (epoch % self.config.eval_every_n_epochs == 0)
                    val_metrics = self._validate(
                        val_loader,
                        edge_index,
                        edge_weight,
                        epoch=epoch,
                        run_ranking_eval=run_eval,
                    )
                    history["val_loss"].append(val_metrics.get("loss", 0))
                
                # Update scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.get("loss", train_metrics["loss"]))
                    else:
                        self.scheduler.step()
                
                # Log metrics
                all_metrics = {
                    f"train/{k}": v for k, v in train_metrics.items()
                }
                all_metrics.update({
                    f"val/{k}": v for k, v in val_metrics.items()
                })
                all_metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
                
                if self.tracker:
                    self.tracker.log_metrics(all_metrics, step=epoch)
                
                # Display metrics table
                display_metrics_table(epoch, train_metrics, val_metrics)
                
                # Callback: epoch end
                for cb in self.callbacks:
                    cb.on_epoch_end(self, epoch, train_metrics, val_metrics)
                
                # Check early stopping
                for cb in self.callbacks:
                    if isinstance(cb, EarlyStopping) and cb.should_stop:
                        progress.update(epoch_task, completed=self.config.max_epochs)
                        break
                else:
                    progress.update(epoch_task, advance=1)
                    continue
                break
        
        # Callback: train end
        for cb in self.callbacks:
            cb.on_train_end(self)
        
        # Load best model
        for cb in self.callbacks:
            if isinstance(cb, ModelCheckpoint):
                cb.load_best(self)
                break
        
        return history
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        edge_index: Optional[torch.Tensor],
        edge_weight: Optional[torch.Tensor],
        progress,
    ) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(train_loader)
        
        batch_task = progress.add_task(
            f"[green]Epoch {self.current_epoch + 1}",
            total=num_batches,
        )
        
        for batch_idx, batch in enumerate(train_loader):
            # Callback: batch start
            for cb in self.callbacks:
                cb.on_batch_start(self, batch_idx)
            
            # Move batch to device
            if len(batch) == 3:
                users, pos_items, neg_items = batch
                users = users.to(self.device)
                pos_items = pos_items.to(self.device)
                neg_items = neg_items.to(self.device)
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")
            
            # Forward pass
            self.optimizer.zero_grad()
            
            loss = self.model.compute_loss(
                users=users,
                pos_items=pos_items,
                neg_items=neg_items,
                edge_index=edge_index,
                edge_weight=edge_weight,
                reg_weight=self.config.reg_weight,
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
            
            self.optimizer.step()
            
            # Track loss
            batch_loss = loss.item()
            total_loss += batch_loss
            self.global_step += 1
            
            # Callback: batch end
            for cb in self.callbacks:
                cb.on_batch_end(self, batch_idx, batch_loss)
            
            progress.update(batch_task, advance=1)
        
        progress.remove_task(batch_task)
        
        return {"loss": total_loss / num_batches}
    
    @torch.no_grad()
    def _validate(
        self,
        val_loader: DataLoader,
        edge_index: Optional[torch.Tensor],
        edge_weight: Optional[torch.Tensor],
        epoch: int = 0,
        run_ranking_eval: bool = False,
    ) -> dict[str, float]:
        """Validate the model with optional ranking metrics."""
        self.model.eval()
        
        # Cache embeddings once for both loss computation and ranking evaluation
        user_emb = None
        item_emb = None
        if hasattr(self.model, "forward") and edge_index is not None:
            user_emb, item_emb = self.model.forward(edge_index, edge_weight)
        
        total_loss = 0.0
        num_batches = len(val_loader)
        
        for batch in val_loader:
            if len(batch) == 3:
                users, pos_items, neg_items = batch
                users = users.to(self.device)
                pos_items = pos_items.to(self.device)
                neg_items = neg_items.to(self.device)
            else:
                continue
            
            # Use cached embeddings for loss computation if available
            if user_emb is not None and item_emb is not None:
                # Compute loss using cached embeddings
                user_emb_batch = user_emb[users]
                pos_item_emb = item_emb[pos_items]
                neg_item_emb = item_emb[neg_items]
                
                # BPR loss
                pos_scores = (user_emb_batch * pos_item_emb).sum(dim=1)
                neg_scores = (user_emb_batch * neg_item_emb).sum(dim=1)
                loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
            else:
                loss = self.model.compute_loss(
                    users=users,
                    pos_items=pos_items,
                    neg_items=neg_items,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )
            
            total_loss += loss.item()
        
        metrics = {"loss": total_loss / num_batches if num_batches > 0 else 0}
        
        # Run full ranking evaluation if requested and evaluator is available
        if run_ranking_eval and self.evaluator is not None and self.val_ground_truth is not None:
            eval_results = self.evaluator.evaluate(
                model=self.model,
                ground_truth=self.val_ground_truth,
                train_items=self.val_train_items or {},
                num_items=self.num_items or 0,
                edge_index=edge_index,
                edge_weight=edge_weight,
                device=self.device,
                user_emb=user_emb,
                item_emb=item_emb,
            )
            metrics.update(eval_results)
        
        return metrics
    
    def save_checkpoint(self, path: str | Path) -> None:
        """Save training checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str | Path) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
