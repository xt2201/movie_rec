"""
Rich console logging utilities for ML training.

Provides beautiful terminal output with:
- Progress bars for epochs and batches
- Metric tables
- Configuration panels
- Styled log messages
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


# Global console instance
console = Console()


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure Rich logging handler.
    
    Args:
        level: Logging level
        
    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_time=True,
                show_path=False,
            )
        ],
    )
    return logging.getLogger("movie_rec")


def create_progress(transient: bool = False) -> Progress:
    """
    Create a Rich progress bar for training.
    
    Args:
        transient: Whether to remove progress bar when done
        
    Returns:
        Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
        refresh_per_second=10,
    )


def display_metrics_table(
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: Optional[dict[str, float]] = None,
    title: Optional[str] = None,
) -> None:
    """
    Display training metrics in a Rich table.
    
    Args:
        epoch: Current epoch number
        train_metrics: Training metrics
        val_metrics: Validation metrics (optional)
        title: Custom title (default: "Epoch {epoch} Results")
    """
    title = title or f"Epoch {epoch} Results"
    
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Train", style="green", justify="right")
    
    if val_metrics:
        table.add_column("Validation", style="yellow", justify="right")
    
    # Combine all metric keys
    all_keys = set(train_metrics.keys())
    if val_metrics:
        all_keys.update(val_metrics.keys())
    
    for key in sorted(all_keys):
        train_val = f"{train_metrics.get(key, 0):.4f}" if key in train_metrics else "-"
        
        if val_metrics:
            val_val = f"{val_metrics.get(key, 0):.4f}" if key in val_metrics else "-"
            table.add_row(key, train_val, val_val)
        else:
            table.add_row(key, train_val)
    
    console.print(table)
    console.print()


def display_config(config: dict[str, Any], title: str = "Configuration") -> None:
    """
    Display configuration in a Rich panel.
    
    Args:
        config: Configuration dictionary
        title: Panel title
    """
    config_lines = []
    
    for key, value in config.items():
        if isinstance(value, dict):
            config_lines.append(f"[cyan]{key}[/cyan]:")
            for k, v in value.items():
                config_lines.append(f"  [dim]{k}[/dim]: {v}")
        else:
            config_lines.append(f"[cyan]{key}[/cyan]: {value}")
    
    config_str = "\n".join(config_lines)
    console.print(Panel(config_str, title=title, border_style="blue"))
    console.print()


def display_model_summary(model, title: str = "Model Summary") -> None:
    """
    Display model architecture summary.
    
    Args:
        model: PyTorch model
        title: Panel title
    """
    lines = [
        f"[cyan]Model[/cyan]: {model.__class__.__name__}",
        f"[cyan]Parameters[/cyan]: {sum(p.numel() for p in model.parameters()):,}",
        f"[cyan]Trainable[/cyan]: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    ]
    
    if hasattr(model, "num_users"):
        lines.append(f"[cyan]Users[/cyan]: {model.num_users:,}")
    if hasattr(model, "num_items"):
        lines.append(f"[cyan]Items[/cyan]: {model.num_items:,}")
    if hasattr(model, "embedding_dim"):
        lines.append(f"[cyan]Embedding Dim[/cyan]: {model.embedding_dim}")
    
    console.print(Panel("\n".join(lines), title=title, border_style="green"))
    console.print()


class RichLogger:
    """
    Rich-based logger for ML training.
    
    Provides methods for logging with different styles and levels.
    """
    
    def __init__(self, name: str = "movie_rec", level: str = "INFO"):
        """
        Initialize Rich logger.
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = setup_logging(level)
        self.console = console
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def success(self, message: str) -> None:
        """Log success message with green styling."""
        self.console.print(f"[green]✓[/green] {message}")
    
    def step(self, message: str) -> None:
        """Log step message with arrow."""
        self.console.print(f"[blue]→[/blue] {message}")
    
    def metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric."""
        step_str = f"[dim](step {step})[/dim] " if step is not None else ""
        self.console.print(f"{step_str}[cyan]{name}[/cyan]: {value:.4f}")
    
    def metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics on one line."""
        step_str = f"[dim]step {step}[/dim] | " if step is not None else ""
        metric_strs = [f"[cyan]{k}[/cyan]: {v:.4f}" for k, v in metrics.items()]
        self.console.print(step_str + " | ".join(metric_strs))
    
    def separator(self, char: str = "─", length: int = 50) -> None:
        """Print a separator line."""
        self.console.print(f"[dim]{char * length}[/dim]")
    
    def header(self, text: str) -> None:
        """Print a header."""
        self.console.print()
        self.console.rule(f"[bold]{text}[/bold]")
        self.console.print()
    
    def print_table(self, data: dict[str, list], title: Optional[str] = None) -> None:
        """
        Print data as a table.
        
        Args:
            data: Dict mapping column names to values
            title: Table title
        """
        table = Table(title=title, show_header=True)
        
        for col in data.keys():
            table.add_column(col, style="cyan")
        
        # Transpose data to rows
        n_rows = len(next(iter(data.values())))
        for i in range(n_rows):
            row = [str(data[col][i]) for col in data.keys()]
            table.add_row(*row)
        
        self.console.print(table)
