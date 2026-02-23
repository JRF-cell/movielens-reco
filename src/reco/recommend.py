from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .eval import temporal_user_split
from .itemcos import fit_item_cosine, recommend_user


@dataclass(frozen=True)
class Movie:
    movieId: int
    title: str


def load_movies(dataset_dir: str | Path) -> pd.DataFrame:
    dataset_dir = Path(dataset_dir)
    movies_path = dataset_dir / "movies.csv"
    if not movies_path.exists():
        raise FileNotFoundError(f"Missing {movies_path}")
    return pd.read_csv(movies_path)


def recommend_for_user(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    k: int = 10,
    test_ratio: float = 0.2,
    min_user_ratings: int = 10,
    n_seed: int = 20,
    n_neighbors: int = 50,
) -> pd.DataFrame:
    split = temporal_user_split(ratings, test_ratio=test_ratio, min_user_ratings=min_user_ratings)
    train, test = split.train, split.test

    # Vérifier user présent (après filtrage min_user_ratings)
    if user_id not in set(train["userId"].unique()):
        raise ValueError(f"userId={user_id} not found in train split (or too few ratings).")

    model = fit_item_cosine(train, n_neighbors=max(n_neighbors + 1, 10))

    user_train = train[train["userId"] == user_id]
    rec_ids = recommend_user(model, user_train, k=k, n_seed=n_seed, n_neighbors=n_neighbors)

    # Map ids -> titles
    rec_df = pd.DataFrame({"movieId": rec_ids})
    out = rec_df.merge(movies[["movieId", "title"]], on="movieId", how="left")

    # Optionnel: indiquer si un recommandé est dans le test (sanity)
    test_set = set(test[test["userId"] == user_id]["movieId"].astype(int).tolist())
    out["in_user_test"] = out["movieId"].apply(lambda x: int(x) in test_set)

    return out
