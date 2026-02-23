from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class ItemCosineModel:
    item_ids: np.ndarray                # idx -> movieId
    item_to_idx: dict[int, int]         # movieId -> idx
    neighbors_idx: np.ndarray           # shape: (n_items, n_neighbors+1) indices in [0..n_items)
    neighbors_sim: np.ndarray           # shape: (n_items, n_neighbors+1) similarity in [0..1]


def fit_item_cosine(train: pd.DataFrame, n_neighbors: int = 50) -> ItemCosineModel:
    user_ids = pd.Index(train["userId"].unique()).sort_values()
    item_ids = pd.Index(train["movieId"].unique()).sort_values()

    user_map = pd.Series(range(len(user_ids)), index=user_ids)
    item_map = pd.Series(range(len(item_ids)), index=item_ids)

    rows = train["userId"].map(user_map).to_numpy()
    cols = train["movieId"].map(item_map).to_numpy()
    vals = train["rating"].to_numpy(dtype=float)

    # user-item ratings
    user_item = sparse.csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    # item-user ratings (for item similarity)
    item_user = user_item.T.tocsr()

    n_items = item_user.shape[0]
    k = min(n_neighbors + 1, n_items)  # +1 to include self neighbor

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k, n_jobs=-1)
    nn.fit(item_user)

    # Precompute neighbors for ALL items once
    dists, neigh = nn.kneighbors(item_user, n_neighbors=k, return_distance=True)
    sims = 1.0 - dists  # cosine distance -> similarity

    neigh = neigh.astype(np.int32, copy=False)
    sims = sims.astype(np.float32, copy=False)

    item_ids_arr = item_ids.to_numpy()
    item_to_idx = {int(mid): int(i) for i, mid in enumerate(item_ids_arr)}

    return ItemCosineModel(
        item_ids=item_ids_arr,
        item_to_idx=item_to_idx,
        neighbors_idx=neigh,
        neighbors_sim=sims,
    )


def recommend_user(
    model: ItemCosineModel,
    user_train: pd.DataFrame,
    k: int = 10,
    n_seed: int = 20,
    n_neighbors: int = 50,
) -> list[int]:
    if user_train.empty:
        return []

    # Seeds = dernières interactions
    if "timestamp" in user_train.columns:
        seeds = user_train.sort_values("timestamp", ascending=False).head(n_seed)
    else:
        seeds = user_train.tail(n_seed)

    seen = set(user_train["movieId"].astype(int).tolist())
    scores: dict[int, float] = {}

    max_k = model.neighbors_idx.shape[1]
    req = min(n_neighbors + 1, max_k)

    for _, r in seeds.iterrows():
        mid = int(r["movieId"])
        rating = float(r["rating"])
        idx = model.item_to_idx.get(mid)
        if idx is None:
            continue

        neigh_idx = model.neighbors_idx[idx, :req]
        neigh_sim = model.neighbors_sim[idx, :req]

        for j, sim in zip(neigh_idx, neigh_sim, strict=True):
            cand_mid = int(model.item_ids[int(j)])
            if cand_mid == mid or cand_mid in seen:
                continue
            if sim <= 0.0:
                continue
            # rating-weighted similarity aggregation
            scores[cand_mid] = scores.get(cand_mid, 0.0) + float(sim) * rating

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [mid for mid, _ in ranked[:k]]