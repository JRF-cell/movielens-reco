from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    test: pd.DataFrame


def temporal_user_split(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    min_user_ratings: int = 10,
) -> Split:
    """
    Split temporel par utilisateur:
    - on trie par timestamp
    - on met les dernières interactions en test
    - on garde uniquement les utilisateurs avec >= min_user_ratings
    """
    if "timestamp" not in ratings.columns:
        raise ValueError("ratings must include a 'timestamp' column")

    # Filtrer users peu actifs
    counts = ratings.groupby("userId").size()
    keep_users = counts[counts >= min_user_ratings].index
    df = ratings[ratings["userId"].isin(keep_users)].copy()

    train_parts = []
    test_parts = []

    for _uid, g in df.groupby("userId", sort=False):
        g = g.sort_values("timestamp")
        n = len(g)
        n_test = max(1, int(math.floor(n * test_ratio)))
        n_train = n - n_test
        if n_train < 1:
            continue
        train_parts.append(g.iloc[:n_train])
        test_parts.append(g.iloc[n_train:])

    train = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0].copy()
    test = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0].copy()
    return Split(train=train, test=test)


def popularity_ranking(train: pd.DataFrame) -> np.ndarray:
    """Classement global des films par popularité (nb de notes) sur train."""
    pop = train.groupby("movieId").size().sort_values(ascending=False)
    return pop.index.to_numpy()


def _dcg(rels: np.ndarray) -> float:
    # rels: 1/0
    if rels.size == 0:
        return 0.0
    denom = np.log2(np.arange(2, rels.size + 2))
    return float((rels / denom).sum())


def precision_recall_ndcg_at_k(
    recommended: list[int],
    relevant_set: set[int],
    k: int,
) -> tuple[float, float, float]:
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0, 0.0, 0.0

    hits = np.array([1.0 if mid in relevant_set else 0.0 for mid in rec_k], dtype=float)
    precision = float(hits.mean())
    recall = float(hits.sum() / max(1, len(relevant_set)))

    dcg = _dcg(hits)
    ideal = np.ones(min(k, len(relevant_set)), dtype=float)
    idcg = _dcg(ideal)
    ndcg = float(dcg / idcg) if idcg > 0 else 0.0
    return precision, recall, ndcg


def eval_popularity(
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
) -> dict[str, float]:
    """
    Évalue une baseline popularité:
    - pour chaque user: recommander les films les plus populaires non vus en train
    - mesurer P@K, R@K, NDCG@K + coverage
    """
    ranking = popularity_ranking(train)

    # sets train/test par user
    train_seen = train.groupby("userId")["movieId"].apply(set).to_dict()
    test_items = test.groupby("userId")["movieId"].apply(set).to_dict()

    precisions, recalls, ndcgs = [], [], []
    all_recommended = set()

    for _uid, rel_set in test_items.items():
        if not rel_set:
            continue
        seen = train_seen.get(_uid, set())

        # Construire top-k en filtrant les vus
        recs = []
        for mid in ranking:
            if mid not in seen:
                recs.append(int(mid))
                if len(recs) >= k:
                    break

        p, r, n = precision_recall_ndcg_at_k(recs, rel_set, k)
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(n)
        all_recommended.update(recs)

    n_users_eval = float(len(precisions))
    n_items_train = float(train["movieId"].nunique()) if len(train) else 0.0
    coverage = float(len(all_recommended) / n_items_train) if n_items_train else 0.0

    return {
        "users_eval": n_users_eval,
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "coverage": coverage,
    }
