#!/usr/bin/env python
"""
Multi-Model Benchmark Script

Train all recommendation models on MovieLens dataset and consolidate results.

Usage:
    python scripts/benchmark.py --data movielens_1m --epochs 100
    python scripts/benchmark.py --models lightgcn,ngcf,ncf
    python scripts/benchmark.py --models all --device cuda
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from rich.console import Console
from rich.table import Table

console = Console()


# Available models configuration
MODELS_CONFIG = {
    "lightgcn": {
        "type": "graph",
        "gpu": True,
        "config_overrides": {},
    },
    "ngcf": {
        "type": "graph", 
        "gpu": True,
        "config_overrides": {},
    },
    "ncf": {
        "type": "neural",
        "gpu": True,
        "config_overrides": {},
    },
    "svd": {
        "type": "traditional",
        "gpu": False,
        "config_overrides": {},
    },
    "item_cf": {
        "type": "traditional",
        "gpu": False,
        "config_overrides": {},
    },
    "hybrid": {
        "type": "hybrid",
        "gpu": True,
        "config_overrides": {},
        # Use ALL models for ensemble diversity
        "base_models": ["lightgcn", "ngcf", "ncf", "svd", "item_cf"],
    },
}

DEFAULT_MODELS = ["lightgcn", "ngcf", "ncf"]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-Model Benchmark")
    parser.add_argument(
        "--models",
        type=str,
        default="lightgcn,ngcf,ncf",
        help="Comma-separated list of models or 'all' (default: lightgcn,ngcf,ncf)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="movielens_1m",
        help="Dataset config name (default: movielens_1m)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max training epochs (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu/mps)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Evaluate every N epochs (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Base checkpoint directory (default: checkpoints)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )
    return parser.parse_args()


def get_models_to_train(models_str: str) -> list[str]:
    """Parse models argument and return list of model names."""
    if models_str.lower() == "all":
        return list(MODELS_CONFIG.keys())
    
    models = [m.strip().lower() for m in models_str.split(",")]
    
    # Validate
    for model in models:
        if model not in MODELS_CONFIG:
            console.print(f"[red]Unknown model: {model}[/red]")
            console.print(f"Available: {list(MODELS_CONFIG.keys())}")
            sys.exit(1)
    
    return models


def train_gradient_model(
    model_name: str,
    data_name: str,
    epochs: int,
    device: str,
    eval_every: int,
    checkpoint_dir: Path,
    use_wandb: bool,
    use_mlflow: bool,
) -> dict[str, float]:
    """Train a gradient-based model (LightGCN, NGCF, NCF) using Hydra."""
    import subprocess
    import json
    
    model_checkpoint_dir = checkpoint_dir / model_name
    model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "-m", "src.train",
        f"model={model_name}",
        f"data={data_name}",
        f"device={device}",
        f"trainer.max_epochs={epochs}",
        f"trainer.eval_every_n_epochs={eval_every}",
        "trainer.weight_decay=0",
        f"paths.checkpoint_dir={model_checkpoint_dir}",
    ]
    
    if not use_wandb:
        cmd.append("logger.use_wandb=false")
    if not use_mlflow:
        cmd.append("logger.use_mlflow=false")
    
    console.print(f"\n[cyan]Running:[/cyan] {' '.join(cmd)}")
    
    # Run training
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        console.print(f"[red]Training failed for {model_name}[/red]")
        return {}
    
    # Read results from JSON file
    results_path = model_checkpoint_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        return results
    
    console.print(f"[yellow]Warning: results.json not found for {model_name}[/yellow]")
    return {"status": "completed"}


def train_traditional_model(
    model_name: str,
    data_name: str,
    checkpoint_dir: Path,
) -> dict[str, float]:
    """Train a traditional model (SVD, ItemCF)."""
    from src.data import MovieLensDataModule
    from src.evaluation import Evaluator
    from src.evaluation.sampled_evaluator import sampled_evaluate
    
    console.print(f"\n[cyan]Training {model_name}...[/cyan]")
    
    # Load data
    data_dir = Path("data")
    
    # Use hydra config loading
    from omegaconf import OmegaConf
    data_cfg = OmegaConf.load(f"configs/data/{data_name}.yaml")
    
    data_module = MovieLensDataModule(
        data_dir=str(data_dir),
        cfg=data_cfg,
        seed=42,
    )
    data_module.setup()
    
    model_checkpoint_dir = checkpoint_dir / model_name
    model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Build model
    if model_name == "svd":
        from src.models.traditional import SVDRecommender
        
        # Use GPU for SVD training
        svd_device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"[cyan]SVD training on: {svd_device}[/cyan]")
        
        model = SVDRecommender(
            num_users=data_module.num_users,
            num_items=data_module.num_items,
            n_factors=64,
            n_epochs=100,
            lr=0.05,
            reg=0.0001,
            batch_size=65536,  # Large batch for GPU
            device=svd_device,
        )
        
        # Train with PyTorch GPU implementation
        train_df = data_module.data_split.train_df
        model.fit(
            user_ids=train_df["user_idx"].values,
            item_ids=train_df["item_idx"].values,
            verbose=True,
        )
            
    elif model_name == "item_cf":
        from src.models.traditional import ItemBasedCF
        model = ItemBasedCF(
            num_users=data_module.num_users,
            num_items=data_module.num_items,
            k_neighbors=50,
        )
        
        train_df = data_module.data_split.train_df
        model.fit(
            user_ids=train_df["user_idx"].values,
            item_ids=train_df["item_idx"].values,
            ratings=train_df["rating"].values,
        )
    else:
        console.print(f"[red]Unknown traditional model: {model_name}[/red]")
        return {}
    
    # Save model
    model.save(model_checkpoint_dir / "final_model.pt")
    
    # Evaluate
    # Auto-detect evaluation protocol
    split_strategy = data_cfg.get("split", {}).get("strategy", "random")
    ground_truth, train_items = data_module.get_evaluation_data("test")
    
    if split_strategy == "leave_one_out":
        console.print("[yellow]Using sampled evaluation (LOO: 99 neg + 1 pos)[/yellow]")
        results = sampled_evaluate(
            model=model,
            ground_truth=ground_truth,
            train_items=train_items,
            num_items=data_module.num_items,
            num_neg_samples=99,
            k_values=[5, 10, 20],
            edge_index=None,
            edge_weight=None,
            device=torch.device("cpu"),
        )
    else:
        console.print("[yellow]Using full-rank evaluation[/yellow]")
        evaluator = Evaluator(
            k_values=[5, 10, 20],
            metrics=["precision", "recall", "ndcg", "hit_rate"],
        )
        results = evaluator.evaluate(
            model=model,
            ground_truth=ground_truth,
            train_items=train_items,
            num_items=data_module.num_items,
            device=torch.device("cpu"),
        )
        evaluator.print_results(results)
    
    return results


def train_hybrid_model(
    data_name: str,
    checkpoint_dir: Path,
    device: str,
    base_models: list[str],
) -> dict[str, float]:
    """Train a hybrid ensemble model using pretrained base models."""
    from src.data import MovieLensDataModule
    from src.evaluation import Evaluator
    from src.evaluation.sampled_evaluator import sampled_evaluate
    from src.models import LightGCN, NCF, NGCF, HybridEnsemble
    
    console.print(f"\n[cyan]Training hybrid with base models: {base_models}[/cyan]")
    
    # Load data
    data_dir = Path("data")
    from omegaconf import OmegaConf
    data_cfg = OmegaConf.load(f"configs/data/{data_name}.yaml")
    
    data_module = MovieLensDataModule(
        data_dir=str(data_dir),
        cfg=data_cfg,
        seed=42,
    )
    data_module.setup()
    
    # Load pretrained models
    models_dict = {}
    device_torch = torch.device(device)
    
    for model_name in base_models:
        model_path = checkpoint_dir / model_name / "final_model.pt"
        if not model_path.exists():
            console.print(f"[yellow]Warning: {model_name} checkpoint not found, skipping[/yellow]")
            continue
        
        # Load model based on type
        if model_name == "lightgcn":
            model = LightGCN(
                num_users=data_module.num_users,
                num_items=data_module.num_items,
                embedding_dim=64,
                num_layers=3,
                device=device,
            )
        elif model_name == "ncf":
            model = NCF(
                num_users=data_module.num_users,
                num_items=data_module.num_items,
                embedding_dim=64,
                mf_dim=64,
                mlp_layers=[128, 64, 32, 16],
                device=device,
            )
        elif model_name == "ngcf":
            model = NGCF(
                num_users=data_module.num_users,
                num_items=data_module.num_items,
                embedding_dim=64,
                num_layers=3,
                device=device,
            )
        elif model_name == "item_cf":
            from src.models.traditional import ItemBasedCF
            model = ItemBasedCF(
                num_users=data_module.num_users,
                num_items=data_module.num_items,
                k_neighbors=50,
                device="cpu",  # Force CPU to avoid GPU OOM in hybrid
            )
            # ItemCF uses pickle format
            model.load(model_path)
            models_dict[model_name] = model
            console.print(f"  Loaded {model_name}")
            continue
        elif model_name == "svd":
            from src.models.traditional import SVDRecommender
            # Force CPU for Hybrid to avoid OOM/Conflicts
            model = SVDRecommender(
                num_users=data_module.num_users,
                num_items=data_module.num_items,
                n_factors=64,
                device="cpu",
            )
            # SVD uses torch checkpoint with numpy arrays
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model = model.to("cpu")
            model.eval()
            models_dict[model_name] = model
            console.print(f"  Loaded {model_name}")
            continue
        else:
            console.print(f"[yellow]Unknown base model: {model_name}[/yellow]")
            continue
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device_torch)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device_torch)
        model.eval()
        models_dict[model_name] = model
        console.print(f"  Loaded {model_name}")
    
    if len(models_dict) < 2:
        console.print("[red]Need at least 2 models for hybrid ensemble[/red]")
        return {}
    
    # Get edge_index for graph models (keep on CPU)
    edge_index = data_module.edge_index.to("cpu") if hasattr(data_module, 'edge_index') else None
    
    # Create hybrid ensemble (CPU)
    hybrid = HybridEnsemble(
        num_users=data_module.num_users,
        num_items=data_module.num_items,
        models=models_dict,
        weights={name: 1.0 / len(models_dict) for name in models_dict},
        fusion_method="weighted_average",
        k_rrf=20,  # Lower k = more emphasis on top-ranked items
        device="cpu",
        edge_index=edge_index,
    )
    
    # Save hybrid config
    hybrid_dir = checkpoint_dir / "hybrid"
    hybrid_dir.mkdir(parents=True, exist_ok=True)
    
    # Get validation data for weight optimization
    val_ground_truth, val_train_items = data_module.get_evaluation_data("val")
    val_users = list(val_ground_truth.keys())
    # Optimize weights on validation set
    console.print("[cyan]Optimizing ensemble weights on validation set...[/cyan]")
    
    # Get validation data for weight optimization
    val_ground_truth, val_train_items = data_module.get_evaluation_data("val")
    val_users = list(val_ground_truth.keys())
    
    best_weights = hybrid.optimize_weights(
        val_users=val_users,
        ground_truth=val_ground_truth,
        train_items=val_train_items,
        k=10,
        weight_steps=11,
    )
    hybrid.set_weights(best_weights)
    console.print(f"[green]Optimized weights: {best_weights}[/green]")
    
    # Save optimized hybrid config
    hybrid.save(hybrid_dir / "final_model.pt")
    
    # Evaluate on test set
    # Auto-detect evaluation protocol
    split_strategy = data_cfg.get("split", {}).get("strategy", "random")
    ground_truth, train_items = data_module.get_evaluation_data("test")
    
    if split_strategy == "leave_one_out":
        console.print("[yellow]Using sampled evaluation (LOO: 99 neg + 1 pos)[/yellow]")
        results = sampled_evaluate(
            model=model,
            ground_truth=ground_truth,
            train_items=train_items,
            num_items=data_module.num_items,
            num_neg_samples=99,
            k_values=[5, 10, 20],
            edge_index=None,
            edge_weight=None,
            device=torch.device("cpu"),
        )
    else:
        console.print("[yellow]Using full-rank evaluation[/yellow]")
        evaluator = Evaluator(
            k_values=[5, 10, 20],
            metrics=["precision", "recall", "ndcg", "hit_rate"],
        )
        results = evaluator.evaluate(
            model=model,
            ground_truth=ground_truth,
            train_items=train_items,
            num_items=data_module.num_items,
            device=torch.device("cpu"),
        )
        evaluator.print_results(results)
    
    return results


def save_results(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save benchmark results to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"
    
    # Check if file exists to determine if we need headers
    file_exists = csv_path.exists()
    
    if results:
        fieldnames = list(results[0].keys())
        
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(results)
    
    console.print(f"\n[green]Results saved to {csv_path}[/green]")


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    """Print comparison table of all model results."""
    if not results:
        return
    
    table = Table(title="Benchmark Results", show_header=True)
    
    # Add columns
    table.add_column("Model", style="cyan")
    table.add_column("NDCG@10", justify="right")
    table.add_column("Hit Rate@10", justify="right")
    table.add_column("Recall@10", justify="right")
    table.add_column("Precision@10", justify="right")
    table.add_column("Status", style="green")
    
    for r in results:
        table.add_row(
            r.get("model", ""),
            f"{r.get('ndcg_at_10', 0):.4f}",
            f"{r.get('hit_rate_at_10', 0):.4f}",
            f"{r.get('recall_at_10', 0):.4f}",
            f"{r.get('precision_at_10', 0):.4f}",
            r.get("status", "unknown"),
        )
    
    console.print(table)


def main():
    """Main benchmark function."""
    args = parse_args()
    
    console.rule("[bold blue]Multi-Model Benchmark[/bold blue]")
    console.print(f"[cyan]Dataset:[/cyan] {args.data}")
    console.print(f"[cyan]Device:[/cyan] {args.device}")
    console.print(f"[cyan]Max Epochs:[/cyan] {args.epochs}")
    
    models = get_models_to_train(args.models)
    console.print(f"[cyan]Models:[/cyan] {models}")
    
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    
    all_results = []
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    for model_name in models:
        console.rule(f"[bold]Training {model_name.upper()}[/bold]")
        
        model_config = MODELS_CONFIG[model_name]
        
        try:
            if model_config["type"] in ["graph", "neural"]:
                # GPU-based gradient training
                results = train_gradient_model(
                    model_name=model_name,
                    data_name=args.data,
                    epochs=args.epochs,
                    device=args.device if model_config["gpu"] else "cpu",
                    eval_every=args.eval_every,
                    checkpoint_dir=checkpoint_dir,
                    use_wandb=not args.no_wandb,
                    use_mlflow=not args.no_mlflow,
                )
            elif model_config["type"] == "hybrid":
                # Hybrid ensemble - requires pretrained base models
                results = train_hybrid_model(
                    data_name=args.data,
                    checkpoint_dir=checkpoint_dir,
                    device=args.device,
                    base_models=model_config.get("base_models", ["lightgcn", "ncf"]),
                )
            else:
                # Traditional CPU-based training
                results = train_traditional_model(
                    model_name=model_name,
                    data_name=args.data,
                    checkpoint_dir=checkpoint_dir,
                )
            
            if results:
                results["model"] = model_name
                results["dataset"] = args.data
                results["timestamp"] = timestamp
                results["status"] = "completed"
                all_results.append(results)
                
        except Exception as e:
            console.print(f"[red]Error training {model_name}: {e}[/red]")
            all_results.append({
                "model": model_name,
                "dataset": args.data,
                "timestamp": timestamp,
                "status": f"failed: {str(e)[:50]}",
            })
    
    # Save and display results
    console.rule("[bold green]Benchmark Complete[/bold green]")
    
    if all_results:
        save_results(all_results, output_dir)
        print_comparison_table(all_results)
    
    # Generate markdown table
    md_path = output_dir / "comparison_table.md"
    with open(md_path, "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(f"Dataset: {args.data}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("| Model | NDCG@10 | Hit Rate@10 | Recall@10 | Precision@10 |\n")
        f.write("|-------|---------|-------------|-----------|-------------|\n")
        for r in all_results:
            f.write(f"| {r.get('model', '')} | "
                   f"{r.get('ndcg_at_10', 0):.4f} | "
                   f"{r.get('hit_rate_at_10', 0):.4f} | "
                   f"{r.get('recall_at_10', 0):.4f} | "
                   f"{r.get('precision_at_10', 0):.4f} |\n")
    
    console.print(f"[green]Markdown table saved to {md_path}[/green]")


if __name__ == "__main__":
    main()
