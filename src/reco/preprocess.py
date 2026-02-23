from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class RatingsData:
    ratings: pd.DataFrame
    user_index: pd.Index
    item_index: pd.Index
    matrix: sparse.csr_matrix


def load_ratings_csv(dataset_dir: str | Path) -> pd.DataFrame:
    dataset_dir = Path(dataset_dir)
    ratings_path = dataset_dir / "ratings.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing {ratings_path}")
    df = pd.read_csv(ratings_path)
    # Columns expected: userId,movieId,rating,timestamp
    return df


def build_user_item_matrix(ratings: pd.DataFrame) -> RatingsData:
    # Remap ids to 0..n-1 indices (compact)
    user_ids = pd.Index(ratings["userId"].unique()).sort_values()
    item_ids = pd.Index(ratings["movieId"].unique()).sort_values()

    user_map = pd.Series(range(len(user_ids)), index=user_ids)
    item_map = pd.Series(range(len(item_ids)), index=item_ids)

    rows = ratings["userId"].map(user_map).to_numpy()
    cols = ratings["movieId"].map(item_map).to_numpy()
    vals = ratings["rating"].to_numpy(dtype=float)

    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    return RatingsData(ratings=ratings, user_index=user_ids, item_index=item_ids, matrix=mat)


def basic_stats(rd: RatingsData) -> dict[str, float]:
    n_users, n_items = rd.matrix.shape
    nnz = rd.matrix.nnz
    density = nnz / (n_users * n_items) if n_users and n_items else 0.0
    return {
        "n_users": float(n_users),
        "n_items": float(n_items),
        "n_ratings": float(nnz),
        "density": float(density),
        "min_rating": float(rd.ratings["rating"].min()),
        "max_rating": float(rd.ratings["rating"].max()),
        "mean_rating": float(rd.ratings["rating"].mean()),
    }
