from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


@dataclass(frozen=True)
class SVDModel:
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_to_idx: dict[int, int]
    item_to_idx: dict[int, int]
    user_factors: np.ndarray  # shape (n_users, k)
    item_factors: np.ndarray  # shape (n_items, k)
    global_mean: float


def fit_svd(train: pd.DataFrame, n_components: int = 50, random_state: int = 42) -> SVDModel:
    user_ids = pd.Index(train["userId"].unique()).sort_values()
    item_ids = pd.Index(train["movieId"].unique()).sort_values()

    user_map = pd.Series(range(len(user_ids)), index=user_ids)
    item_map = pd.Series(range(len(item_ids)), index=item_ids)

    rows = train["userId"].map(user_map).to_numpy()
    cols = train["movieId"].map(item_map).to_numpy()

    global_mean = float(train["rating"].mean())
    vals = (train["rating"].to_numpy(dtype=float) - global_mean)

    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_ids)))

    # TruncatedSVD requires n_components < min(n_users, n_items)
    max_k = max(2, min(X.shape) - 1)
    k = min(n_components, max_k)

    svd = TruncatedSVD(n_components=k, random_state=random_state)
    user_factors = svd.fit_transform(X)          # U * Sigma  (n_users, k)
    item_factors = svd.components_.T             # V          (n_items, k)

    user_ids_arr = user_ids.to_numpy()
    item_ids_arr = item_ids.to_numpy()

    return SVDModel(
        user_ids=user_ids_arr,
        item_ids=item_ids_arr,
        user_to_idx={int(u): int(i) for i, u in enumerate(user_ids_arr)},
        item_to_idx={int(m): int(i) for i, m in enumerate(item_ids_arr)},
        user_factors=user_factors.astype(np.float32, copy=False),
        item_factors=item_factors.astype(np.float32, copy=False),
        global_mean=global_mean,
    )


def recommend_user_svd(
    model: SVDModel,
    user_train: pd.DataFrame,
    k: int = 10,
) -> list[int]:
    if user_train.empty:
        return []

    uid = int(user_train["userId"].iloc[0])
    uidx = model.user_to_idx.get(uid)
    if uidx is None:
        return []

    uvec = model.user_factors[uidx]  # (k,)
    scores = model.item_factors @ uvec  # (n_items,)

    # Exclude seen items
    seen = set(user_train["movieId"].astype(int).tolist())
    for mid in seen:
        j = model.item_to_idx.get(mid)
        if j is not None:
            scores[j] = -np.inf

    if k <= 0:
        return []

    k = min(k, scores.shape[0])
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    return [int(model.item_ids[j]) for j in top_idx]