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

Train a LightGCN model:

```bash
python -m src.train model=lightgcn
```

Train with custom hyperparameters:

```bash
python -m src.train model=lightgcn \
    model.embedding_dim=128 \
    model.num_layers=4 \
    trainer.max_epochs=100 \
    trainer.learning_rate=0.001
```

Train NCF model:

```bash
python -m src.train model=ncf
```

### Hyperparameter Sweeps

```bash
python -m src.train --multirun \
    model=lightgcn,ngcf,ncf \
    trainer.learning_rate=0.001,0.0001
```

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

## 📊 Metrics

The system supports the following evaluation metrics:

- **Precision@K** - Fraction of relevant items in top-K
- **Recall@K** - Fraction of relevant items retrieved
- **NDCG@K** - Normalized Discounted Cumulative Gain
- **Hit Rate@K** - Whether any relevant item is in top-K
- **MAP@K** - Mean Average Precision
- **MRR@K** - Mean Reciprocal Rank
- **Coverage** - Catalog coverage

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

## 📚 References

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
