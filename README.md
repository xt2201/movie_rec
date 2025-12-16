# Movie Recommendation Unified

A production-ready movie recommendation system combining multiple state-of-the-art approaches in a unified, extensible architecture.

## 🌟 Features

### Recommendation Approaches

| Model | Type | Description |
|-------|------|-------------|
| **LightGCN** | Graph Neural Network | Simplified GCN that learns user/item embeddings through graph convolution |
| **NGCF** | Graph Neural Network | Neural Graph Collaborative Filtering with feature transformation |
| **NCF** | Neural Network | Neural Collaborative Filtering combining GMF and MLP |
| **GMF** | Neural Network | Generalized Matrix Factorization |
| **MLP** | Neural Network | Multi-Layer Perceptron for learning non-linear interactions |

### Infrastructure

- **PyTorch 2.0+** - Core deep learning framework
- **PyTorch Geometric** - Graph neural network operations
- **FAISS** - Fast similarity search for real-time inference
- **MLflow** - Experiment tracking and model registry
- **Weights & Biases** - Training visualization and hyperparameter sweeps
- **Rich** - Beautiful console logging and progress bars
- **Hydra** - Hierarchical configuration management
- **Flask** - REST API serving

## 📁 Project Structure

```
movie_rec_unified/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config with defaults
│   ├── data/
│   │   └── movielens.yaml     # Dataset configuration
│   ├── model/
│   │   ├── lightgcn.yaml      # LightGCN hyperparameters
│   │   ├── ngcf.yaml          # NGCF hyperparameters
│   │   ├── ncf.yaml           # NCF hyperparameters
│   │   └── ...
│   ├── trainer/
│   │   └── default.yaml       # Training configuration
│   └── logger/
│       └── wandb_mlflow.yaml  # Logging configuration
│
├── data/                       # Data directory
│   └── movielens-small/       # MovieLens 100K dataset
│       ├── ratings.csv
│       ├── movies.csv
│       └── ...
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── train.py               # Main training script
│   │
│   ├── api/                   # Flask API
│   │   ├── __init__.py
│   │   ├── app.py             # Flask application factory
│   │   └── inference.py       # Inference service
│   │
│   ├── data/                  # Data processing
│   │   ├── __init__.py
│   │   ├── datamodule.py      # Main data module
│   │   ├── preprocessor.py    # Data preprocessing
│   │   ├── graph_builder.py   # Graph construction
│   │   └── negative_sampler.py # Negative sampling
│   │
│   ├── models/                # Recommendation models
│   │   ├── __init__.py
│   │   ├── base.py            # BaseRecommender ABC
│   │   ├── graph/             # Graph-based models
│   │   │   ├── __init__.py
│   │   │   ├── lightgcn.py
│   │   │   └── ngcf.py
│   │   └── neural/            # Neural network models
│   │       ├── __init__.py
│   │       └── ncf.py
│   │
│   ├── retrieval/             # Fast retrieval
│   │   ├── __init__.py
│   │   └── faiss_index.py     # FAISS indexing
│   │
│   ├── training/              # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py         # Training loop
│   │   ├── losses.py          # Loss functions
│   │   └── callbacks.py       # Training callbacks
│   │
│   ├── evaluation/            # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── metrics.py         # Ranking metrics
│   │   └── evaluator.py       # Evaluation orchestrator
│   │
│   ├── utils/                 # Utilities
│   │   ├── __init__.py
│   │   ├── experiment_tracker.py
│   │   └── rich_logging.py
│   │
│   ├── templates/             # HTML templates
│   └── static/                # CSS/JS assets
│
├── pyproject.toml             # Project configuration
├── run_server.py              # API server entry point
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/xt2201/movie_rec.git
cd movie_rec

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### Environment Setup

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```bash
# Weights & Biases (required for experiment tracking)
WANDB_API_KEY=your_wandb_api_key_here

# MLflow (optional)
MLFLOW_TRACKING_URI=./mlruns
```

### Download Data

Download the MovieLens Small dataset and place it in `data/movielens-small/`:

```bash
wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip ml-latest-small.zip
mv ml-latest-small/* data/movielens-small/
```

### Training

#### 🔷 Graph Neural Networks

**LightGCN** - Simplified Graph Convolution Network:
```bash
# Basic training
python -m src.train model=lightgcn

# Custom hyperparameters
python -m src.train model=lightgcn \
    model.embedding_dim=128 \
    model.num_layers=4 \
    trainer.max_epochs=100 \
    trainer.learning_rate=0.001

# With specific settings
python -m src.train model=lightgcn \
    model.embedding_dim=64 \
    model.num_layers=3 \
    model.dropout=0.1 \
    trainer.batch_size=2048 \
    trainer.learning_rate=0.001
```

**NGCF** - Neural Graph Collaborative Filtering:
```bash
# Basic training
python -m src.train model=ngcf

# Custom architecture
python -m src.train model=ngcf \
    model.embedding_dim=64 \
    model.hidden_dims=[64,64,64] \
    model.dropout=0.1 \
    model.message_dropout=0.1 \
    trainer.learning_rate=0.0001
```

#### 🔷 Neural Collaborative Filtering

**NCF** - Neural Collaborative Filtering (GMF + MLP):
```bash
# Basic training
python -m src.train model=ncf

# Custom architecture
python -m src.train model=ncf \
    model.mf_dim=64 \
    model.mlp_layers=[128,64,32,16] \
    model.dropout=0.2 \
    model.num_negatives=4 \
    trainer.learning_rate=0.001

# With BCEWithLogits loss
python -m src.train model=ncf \
    model.loss_type=bce \
    trainer.batch_size=1024 \
    trainer.max_epochs=50
```

#### 🔷 Traditional Methods

**SVD** - Singular Value Decomposition (using Surprise):
```bash
# Basic training
python -m src.train model=svd

# Custom factors and regularization
python -m src.train model=svd \
    model.n_factors=100 \
    model.n_epochs=20 \
    model.lr_all=0.005 \
    model.reg_all=0.02

# Unbiased SVD
python -m src.train model=svd \
    model.biased=false \
    model.n_factors=150
```

**Item-based Collaborative Filtering**:
```bash
# Basic training (cosine similarity)
python -m src.train model=item_cf

# With different similarity metrics
python -m src.train model=item_cf \
    model.k=40 \
    model.similarity=pearson

# Adjusted similarity with more neighbors
python -m src.train model=item_cf \
    model.k=50 \
    model.similarity=pearson_baseline \
    model.shrinkage=100
```

#### 🔷 Hybrid Ensemble

**Hybrid Ensemble** - Combines multiple models:
```bash
# Basic hybrid training
python -m src.train model=hybrid_ensemble

# Custom model combination
python -m src.train model=hybrid_ensemble \
    model.models=[ncf,item_cf] \
    model.fusion_method=weighted_average

# With learned weights
python -m src.train model=hybrid_ensemble \
    model.learn_weights=true \
    model.fusion_method=rrf
```

#### 🔷 Hyperparameter Sweeps

**Compare all models:**
```bash
python -m src.train --multirun \
    model=lightgcn,ngcf,ncf,svd,item_cf \
    seed=42,123,456
```

**Grid search for LightGCN:**
```bash
python -m src.train --multirun \
    model=lightgcn \
    model.embedding_dim=32,64,128 \
    model.num_layers=2,3,4 \
    trainer.learning_rate=0.001,0.0001
```

**NCF architecture search:**
```bash
python -m src.train --multirun \
    model=ncf \
    model.mf_dim=32,64 \
    model.mlp_layers='[64,32,16]','[128,64,32,16]' \
    model.dropout=0.1,0.2,0.3
```

**Learning rate optimization:**
```bash
python -m src.train --multirun \
    model=lightgcn,ngcf,ncf \
    trainer.learning_rate=0.0001,0.0005,0.001,0.005
```

#### 🔷 Training Tips & Best Practices

**Device Selection:**
```bash
# Use GPU if available
python -m src.train model=lightgcn device=cuda

# Use MPS (Apple Silicon)
python -m src.train model=ngcf device=mps

# Force CPU
python -m src.train model=ncf device=cpu
```

**Monitoring Training:**
```bash
# Enable both MLflow and Wandb
python -m src.train model=lightgcn \
    logger.use_mlflow=true \
    logger.use_wandb=true \
    logger.wandb.project=movie-rec-exp

# Disable logging for quick tests
python -m src.train model=ncf \
    logger.use_mlflow=false \
    logger.use_wandb=false
```

**Early Stopping:**
```bash
# Custom early stopping patience
python -m src.train model=lightgcn \
    trainer.early_stopping.enabled=true \
    trainer.early_stopping.patience=20 \
    trainer.early_stopping.metric=ndcg@10 \
    trainer.early_stopping.mode=max
```

**Experiment Tracking:**
```bash
# Custom experiment and run names
python -m src.train model=lightgcn \
    experiment_name=lightgcn_experiments \
    run_name=exp_lr0.001_emb128 \
    model.embedding_dim=128
```

#### 📊 Model Comparison Guide

| Model | Training Time | Memory | Cold Start | Accuracy | Interpretability |
|-------|--------------|---------|------------|----------|------------------|
| **LightGCN** | ⚡⚡ Medium | 💾💾 Medium | ❌ Poor | ⭐⭐⭐⭐⭐ Excellent | ⚠️ Low |
| **NGCF** | ⚡ Slow | 💾💾💾 High | ❌ Poor | ⭐⭐⭐⭐ Good | ⚠️ Low |
| **NCF** | ⚡⚡⚡ Fast | 💾 Low | ❌ Poor | ⭐⭐⭐⭐ Good | ⚠️ Medium |
| **SVD** | ⚡⚡⚡⚡ Very Fast | 💾 Low | ❌ Poor | ⭐⭐⭐ Fair | ✅ High |
| **Item-CF** | ⚡⚡⚡⚡ Very Fast | 💾💾 Medium | ✅ Good | ⭐⭐⭐ Fair | ✅ High |
| **Hybrid** | ⚡⚡ Medium | 💾💾 Medium | ✅ Good | ⭐⭐⭐⭐⭐ Excellent | ⚠️ Medium |

**Recommendations by Use Case:**

- **Highest Accuracy**: LightGCN, Hybrid Ensemble
- **Fastest Training**: SVD, Item-CF
- **Cold Start Users**: Item-CF, Hybrid Ensemble
- **Production Serving**: LightGCN (with FAISS), NCF
- **Interpretability**: SVD, Item-CF
- **Best Overall**: Hybrid Ensemble (combines strengths)

### Running the API Server

```bash
# Start the server
python run_server.py --model-dir checkpoints --data-dir data/movielens-small

# Or with specific options
python run_server.py \
    --host 0.0.0.0 \
    --port 5000 \
    --device cuda \
    --debug
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/recommend` | GET/POST | Get recommendations for a user |
| `/api/similar/<item_id>` | GET | Get similar items |
| `/api/batch-recommend` | POST | Batch recommendations |
| `/api/models` | GET | List available models |
| `/api/movie/<movie_id>` | GET | Get movie details |
| `/api/health` | GET | Health check |

#### Example Request

```bash
# Get recommendations
curl "http://localhost:5000/api/recommend?user_id=1&top_k=10"

# Get similar movies
curl "http://localhost:5000/api/similar/1?top_k=5"

# Batch recommendations
curl -X POST "http://localhost:5000/api/batch-recommend" \
    -H "Content-Type: application/json" \
    -d '{"user_ids": [1, 2, 3], "top_k": 10}'
```

## 📊 Evaluation & Metrics

### Supported Metrics

The system supports the following evaluation metrics:

- **Precision@K** - Fraction of relevant items in top-K recommendations
- **Recall@K** - Fraction of all relevant items that were retrieved
- **NDCG@K** - Normalized Discounted Cumulative Gain (position-aware)
- **Hit Rate@K** - Whether any relevant item appears in top-K
- **MAP@K** - Mean Average Precision across all users
- **MRR@K** - Mean Reciprocal Rank of first relevant item
- **Coverage** - Percentage of catalog items recommended

### Evaluation During Training

Metrics are automatically computed during training at validation time:

```bash
# Evaluate at multiple K values
python -m src.train model=lightgcn \
    evaluation.k_values=[5,10,20] \
    evaluation.metrics=['precision','recall','ndcg','hit_rate']
```

### Post-Training Evaluation

Evaluate a trained model on test set:

```bash
# Using Python API
python -c "
from src.models import LightGCN
from src.data import MovieLensDataModule
from src.evaluation import Evaluator

# Load model and data
model = LightGCN.load('checkpoints/lightgcn_best.pt')
data = MovieLensDataModule('data/movielens-small')
data.setup()

# Evaluate
evaluator = Evaluator(k_values=[5, 10, 20])
results = evaluator.evaluate(model, data.test_data)
evaluator.print_results(results)
"
```

## 🔧 Configuration

### Model Configuration

```yaml
# configs/model/lightgcn.yaml
name: lightgcn
type: graph

embedding_dim: 64
num_layers: 3
dropout: 0.0
```

### Training Configuration

```yaml
# configs/trainer/default.yaml
max_epochs: 100
learning_rate: 0.001
batch_size: 2048
weight_decay: 0.0

optimizer:
  name: adam
  betas: [0.9, 0.999]

early_stopping:
  enabled: true
  patience: 10
  metric: ndcg@10
  mode: max
```

### Logging Configuration

```yaml
# configs/logger/wandb_mlflow.yaml
use_wandb: true
use_mlflow: true

wandb:
  project: movie-rec
  tags: [lightgcn, movielens]
```

## 🧪 Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
```

### Type Checking

```bash
mypy src/
```

## � Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python -m src.train model=lightgcn trainer.batch_size=1024

# Use gradient accumulation
python -m src.train model=ngcf \
    trainer.batch_size=512 \
    trainer.accumulate_grad_batches=4
```

**2. PyTorch Geometric Installation Issues**
```bash
# Install with specific CUDA version
pip install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

**3. Wandb Authentication**
```bash
# Login to Wandb
wandb login

# Or set API key in .env
echo "WANDB_API_KEY=your_key_here" >> .env
```

**4. Slow Training on CPU**
```bash
# Use smaller model or dataset
python -m src.train model=ncf \
    model.mlp_layers=[64,32,16] \
    data.train_size=0.5
```

**5. FileNotFoundError for Data**
```bash
# Ensure data directory exists
ls data/movielens-small/ratings.csv

# Re-download if missing
wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip ml-latest-small.zip -d data/movielens-small/
```

## 🚀 Quick Reference

### Essential Commands

```bash
# Train with default settings
python -m src.train model=<MODEL_NAME>

# Override any config parameter
python -m src.train model=lightgcn <key>=<value>

# Run hyperparameter sweep
python -m src.train --multirun model=<MODELS> <param>=<values>

# View config without training
python -m src.train model=lightgcn --cfg job

# Start API server
python run_server.py --model-dir checkpoints --data-dir data/movielens-small
```

### Model Names

- `lightgcn` - LightGCN graph model
- `ngcf` - Neural Graph Collaborative Filtering
- `ncf` - Neural Collaborative Filtering
- `svd` - Singular Value Decomposition
- `item_cf` - Item-based Collaborative Filtering
- `hybrid_ensemble` - Hybrid ensemble model

### Common Config Overrides

```bash
# Training hyperparameters
trainer.max_epochs=<int>
trainer.learning_rate=<float>
trainer.batch_size=<int>

# Model architecture
model.embedding_dim=<int>
model.num_layers=<int>
model.dropout=<float>

# Data settings
data.train_ratio=<float>
data.val_ratio=<float>
data.negative_samples=<int>

# Logging
logger.use_wandb=<bool>
logger.use_mlflow=<bool>

# Device
device=<cpu|cuda|mps>
```

## �📚 References

- [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126)
- [Neural Graph Collaborative Filtering](https://arxiv.org/abs/1905.08108)
- [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)
- [NVIDIA NCF Implementation](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF)

## 📄 License

MIT License

## 🙏 Acknowledgments

- MovieLens dataset from GroupLens Research
- Original implementations from the respective paper authors
- NVIDIA's NCF reference implementation
