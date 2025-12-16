#!/bin/bash
# Download and prepare MovieLens-1M dataset

set -e

DATA_DIR="data"
ML1M_DIR="$DATA_DIR/ml-1m"

echo "📥 Downloading MovieLens-1M dataset..."

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Download MovieLens-1M
cd "$DATA_DIR"
if [ ! -f "ml-1m.zip" ]; then
    wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
    echo "✓ Downloaded ml-1m.zip"
else
    echo "✓ ml-1m.zip already exists"
fi

# Extract
if [ ! -d "ml-1m" ]; then
    unzip -o ml-1m.zip
    echo "✓ Extracted dataset"
else
    echo "✓ ml-1m directory already exists"
fi

cd ..

echo "🔄 Converting .dat files to .csv..."

# Convert to CSV using Python
python scripts/convert_ml1m.py

echo "✅ MovieLens-1M dataset ready!"
echo "📊 Dataset location: $ML1M_DIR"
echo ""
echo "Files created:"
echo "  - ratings.csv"
echo "  - movies.csv"
echo "  - users.csv"
