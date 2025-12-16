"""
Convert MovieLens-1M .dat files to .csv format.
"""
import pandas as pd
from pathlib import Path

def convert_ml1m():
    """Convert MovieLens-1M .dat files to CSV."""
    data_dir = Path("data/ml-1m")
    
    if not data_dir.exists():
        print(f"❌ Error: {data_dir} does not exist")
        return
    
    print("Converting ratings.dat...")
    ratings = pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
    )
    ratings.to_csv(data_dir / "ratings.csv", index=False)
    print(f"✓ Ratings: {len(ratings):,} rows")
    
    print("Converting movies.dat...")
    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        names=["movieId", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )
    movies.to_csv(data_dir / "movies.csv", index=False)
    print(f"✓ Movies: {len(movies):,} rows")
    
    print("Converting users.dat...")
    users = pd.read_csv(
        data_dir / "users.dat",
        sep="::",
        names=["userId", "gender", "age", "occupation", "zip"],
        engine="python",
    )
    users.to_csv(data_dir / "users.csv", index=False)
    print(f"✓ Users: {len(users):,} rows")
    
    print("\n✅ Conversion complete!")
    print(f"📊 Dataset statistics:")
    print(f"   Users: {users['userId'].nunique():,}")
    print(f"   Movies: {movies['movieId'].nunique():,}")
    print(f"   Ratings: {len(ratings):,}")
    print(f"   Sparsity: {100 * (1 - len(ratings) / (users['userId'].nunique() * movies['movieId'].nunique())):.2f}%")

if __name__ == "__main__":
    convert_ml1m()
