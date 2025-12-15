"""
Training callbacks for early stopping, checkpointing, etc.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import torch

from ..utils.rich_logging import console


class Callback(ABC):
    """Base class for training callbacks."""
    
    def on_train_start(self, trainer: "Trainer") -> None:
        """Called at the start of training."""
        pass
    
    def on_train_end(self, trainer: "Trainer") -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_start(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: Optional[dict[str, float]] = None,
    ) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_start(self, trainer: "Trainer", batch_idx: int) -> None:
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(
        self,
        trainer: "Trainer",
        batch_idx: int,
        loss: float,
    ) -> None:
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback.
    
    Stops training when monitored metric stops improving.
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = "min",
        verbose: bool = True,
    ):
        """
        Initialize early stopping.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' (lower is better) or 'max' (higher is better)
            verbose: Whether to print messages
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.best_value: Optional[float] = None
        self.counter = 0
        self.should_stop = False
        
        if mode == "min":
            self.is_better = lambda new, best: new < best - min_delta
        else:
            self.is_better = lambda new, best: new > best + min_delta
    
    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: Optional[dict[str, float]] = None,
    ) -> None:
        # Get metric value
        metrics = {**train_metrics, **(val_metrics or {})}
        
        if self.monitor not in metrics:
            return
        
        current = metrics[self.monitor]
        
        if self.best_value is None:
            self.best_value = current
            return
        
        if self.is_better(current, self.best_value):
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            
            if self.verbose:
                console.print(
                    f"[yellow]EarlyStopping: {self.monitor} did not improve. "
                    f"Counter: {self.counter}/{self.patience}[/yellow]"
                )
            
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    console.print(
                        f"[red]EarlyStopping: Stopping training at epoch {epoch}[/red]"
                    )
    
    def reset(self) -> None:
        """Reset callback state."""
        self.best_value = None
        self.counter = 0
        self.should_stop = False


class ModelCheckpoint(Callback):
    """
    Model checkpointing callback.
    
    Saves model when monitored metric improves.
    """
    
    def __init__(
        self,
        dirpath: str | Path,
        filename: str = "best_model.pt",
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        save_last: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize model checkpoint.
        
        Args:
            dirpath: Directory to save checkpoints
            filename: Checkpoint filename
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_top_k: Number of best models to keep
            save_last: Whether to save the last model
            verbose: Whether to print messages
        """
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.verbose = verbose
        
        self.best_value: Optional[float] = None
        self.best_path: Optional[Path] = None
        
        if mode == "min":
            self.is_better = lambda new, best: new < best
        else:
            self.is_better = lambda new, best: new > best
    
    def on_train_start(self, trainer: "Trainer") -> None:
        self.dirpath.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: Optional[dict[str, float]] = None,
    ) -> None:
        # Get metric value
        metrics = {**train_metrics, **(val_metrics or {})}
        
        if self.monitor not in metrics:
            # Save without monitoring
            if self.save_last:
                self._save_checkpoint(trainer, epoch, "last_model.pt")
            return
        
        current = metrics[self.monitor]
        
        # Check if improved
        if self.best_value is None or self.is_better(current, self.best_value):
            self.best_value = current
            
            # Save best model
            path = self._save_checkpoint(trainer, epoch, self.filename, current)
            self.best_path = path
            
            if self.verbose:
                console.print(
                    f"[green]Checkpoint: Saved best model ({self.monitor}={current:.4f}) "
                    f"to {path}[/green]"
                )
        
        # Save last model
        if self.save_last:
            self._save_checkpoint(trainer, epoch, "last_model.pt")
    
    def _save_checkpoint(
        self,
        trainer: "Trainer",
        epoch: int,
        filename: str,
        metric_value: Optional[float] = None,
    ) -> Path:
        """Save a checkpoint."""
        path = self.dirpath / filename
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
        }
        
        if trainer.scheduler is not None:
            checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()
        
        if metric_value is not None:
            checkpoint[self.monitor] = metric_value
        
        torch.save(checkpoint, path)
        return path
    
    def load_best(self, trainer: "Trainer") -> None:
        """Load the best checkpoint."""
        if self.best_path and self.best_path.exists():
            checkpoint = torch.load(self.best_path, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            
            if self.verbose:
                console.print(f"[green]Loaded best model from {self.best_path}[/green]")


class LearningRateMonitor(Callback):
    """Log learning rate at each epoch."""
    
    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: Optional[dict[str, float]] = None,
    ) -> None:
        lr = trainer.optimizer.param_groups[0]["lr"]
        
        if trainer.tracker:
            trainer.tracker.log_metric("learning_rate", lr, step=epoch)


class GradientClipCallback(Callback):
    """Callback for gradient clipping (if not using trainer's built-in)."""
    
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
    
    def on_batch_end(
        self,
        trainer: "Trainer",
        batch_idx: int,
        loss: float,
    ) -> None:
        torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(),
            self.max_norm,
        )
