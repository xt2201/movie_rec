"""
Unified experiment tracker for MLflow and Weights & Biases.

Provides a single interface for logging to both platforms simultaneously.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union

import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from .rich_logging import RichLogger, console, display_config

# Load environment variables from .env file
load_dotenv()

# Optional imports
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ExperimentTracker:
    """
    Unified experiment tracker for MLflow and W&B.
    
    Handles:
    - Hyperparameter logging
    - Metric tracking
    - Model checkpointing
    - Artifact logging
    """
    
    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        config: Optional[Union[dict, DictConfig]] = None,
        # MLflow settings
        use_mlflow: bool = True,
        mlflow_tracking_uri: Optional[str] = None,
        # Wandb settings
        use_wandb: bool = True,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_tags: Optional[list[str]] = None,
        # Rich settings
        use_rich: bool = True,
        # Model watching (wandb)
        watch_model: bool = False,
        watch_log: str = "gradients",
        watch_freq: int = 100,
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            run_name: Name of this specific run
            config: Configuration dictionary
            use_mlflow: Whether to use MLflow
            mlflow_tracking_uri: MLflow tracking URI
            use_wandb: Whether to use W&B
            wandb_project: W&B project name
            wandb_entity: W&B entity (username/team)
            wandb_tags: Tags for W&B run
            use_rich: Whether to use Rich console logging
            watch_model: Whether to watch model gradients (wandb)
            watch_log: What to log for model watching
            watch_freq: Frequency of model watching
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.config = config
        
        # Validate availability
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_rich = use_rich
        
        if use_mlflow and not MLFLOW_AVAILABLE:
            console.print("[yellow]Warning: MLflow not installed, skipping[/yellow]")
        if use_wandb and not WANDB_AVAILABLE:
            console.print("[yellow]Warning: wandb not installed, skipping[/yellow]")
        
        # MLflow settings
        self.mlflow_tracking_uri = mlflow_tracking_uri
        
        # Wandb settings
        self.wandb_project = wandb_project or experiment_name
        self.wandb_entity = wandb_entity
        self.wandb_tags = wandb_tags or []
        self.watch_model = watch_model
        self.watch_log = watch_log
        self.watch_freq = watch_freq
        
        # State
        self._mlflow_run = None
        self._wandb_run = None
        self._step = 0
        
        # Rich logger
        if self.use_rich:
            self.logger = RichLogger()
    
    def __enter__(self) -> "ExperimentTracker":
        """Start tracking session."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End tracking session."""
        self.finish()
    
    def start(self) -> None:
        """Start MLflow and W&B runs."""
        # Convert config to dict
        config_dict = {}
        if self.config is not None:
            if isinstance(self.config, DictConfig):
                config_dict = OmegaConf.to_container(self.config, resolve=True)
            else:
                config_dict = dict(self.config)
        
        # Start MLflow
        if self.use_mlflow:
            if self.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            mlflow.set_experiment(self.experiment_name)
            self._mlflow_run = mlflow.start_run(run_name=self.run_name)
            
            # Log params
            if config_dict:
                self._log_mlflow_params(config_dict)
        
        # Start W&B
        if self.use_wandb:
            self._wandb_run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=self.run_name,
                config=config_dict,
                tags=self.wandb_tags,
                reinit=True,
            )
        
        # Display config
        if self.use_rich and config_dict:
            display_config(config_dict, title=f"Experiment: {self.experiment_name}")
    
    def _log_mlflow_params(self, params: dict, prefix: str = "") -> None:
        """Recursively log nested params to MLflow."""
        for key, value in params.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                self._log_mlflow_params(value, full_key)
            else:
                # MLflow param values must be strings
                try:
                    mlflow.log_param(full_key, value)
                except Exception:
                    pass  # Skip if param already logged or invalid
    
    def finish(self) -> None:
        """End MLflow and W&B runs."""
        if self.use_wandb and self._wandb_run:
            wandb.finish()
            self._wandb_run = None
        
        if self.use_mlflow and self._mlflow_run:
            mlflow.end_run()
            self._mlflow_run = None
    
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log metrics to all enabled trackers.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Step number (uses internal counter if None)
        """
        step = step if step is not None else self._step
        
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        
        if self.use_wandb:
            wandb.log(metrics, step=step)
        
        self._step = step + 1
    
    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """Log a single metric."""
        self.log_metrics({name: value}, step=step)
    
    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        if self.use_mlflow:
            self._log_mlflow_params(params)
        
        if self.use_wandb:
            wandb.config.update(params)
    
    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model",
        **kwargs,
    ) -> None:
        """
        Log model to trackers.
        
        Args:
            model: PyTorch model to log
            artifact_path: Path for the artifact
            **kwargs: Additional arguments
        """
        if self.use_mlflow:
            mlflow.pytorch.log_model(model, artifact_path)
        
        if self.use_wandb:
            # Save model and log as artifact
            model_path = f"{artifact_path}.pth"
            torch.save(model.state_dict(), model_path)
            
            artifact = wandb.Artifact(artifact_path, type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            
            # Clean up
            Path(model_path).unlink(missing_ok=True)
    
    def log_artifact(self, path: Union[str, Path], artifact_type: str = "data") -> None:
        """Log an artifact file."""
        path = Path(path)
        
        if self.use_mlflow:
            mlflow.log_artifact(str(path))
        
        if self.use_wandb:
            artifact = wandb.Artifact(path.stem, type=artifact_type)
            artifact.add_file(str(path))
            wandb.log_artifact(artifact)
    
    def watch_model_gradients(self, model: torch.nn.Module) -> None:
        """
        Watch model gradients with W&B.
        
        Args:
            model: PyTorch model to watch
        """
        if self.use_wandb and self.watch_model:
            wandb.watch(
                model,
                log=self.watch_log,
                log_freq=self.watch_freq,
            )
    
    def set_summary(self, key: str, value: Any) -> None:
        """Set a summary metric (final value)."""
        if self.use_mlflow:
            # Replace special characters in metric name for MLflow compatibility
            clean_key = key.replace("@", "_at_")
            mlflow.log_metric(f"final_{clean_key}", value)
        
        if self.use_wandb and wandb.run is not None:
            wandb.run.summary[key] = value
    
    def log_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
    ) -> None:
        """Log a table."""
        if self.use_wandb:
            table = wandb.Table(columns=columns, data=data)
            wandb.log({key: table})
    
    @property
    def run_id(self) -> Optional[str]:
        """Get the current run ID."""
        if self.use_mlflow and self._mlflow_run:
            return self._mlflow_run.info.run_id
        if self.use_wandb and self._wandb_run:
            return wandb.run.id
        return None
    
    @property
    def run_url(self) -> Optional[str]:
        """Get the URL to the current run."""
        if self.use_wandb and self._wandb_run:
            return wandb.run.url
        return None


@contextmanager
def track_experiment(
    experiment_name: str,
    run_name: Optional[str] = None,
    config: Optional[dict] = None,
    **kwargs,
):
    """
    Context manager for experiment tracking.
    
    Example:
        with track_experiment("my_experiment", config=cfg) as tracker:
            trainer.fit(model, datamodule)
            tracker.log_metrics({"accuracy": 0.95})
    """
    tracker = ExperimentTracker(
        experiment_name=experiment_name,
        run_name=run_name,
        config=config,
        **kwargs,
    )
    
    try:
        tracker.start()
        yield tracker
    finally:
        tracker.finish()
