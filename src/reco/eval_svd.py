from __future__ import annotations

import numpy as np
import pandas as pd

from .eval import precision_recall_ndcg_at_k
from .svd import fit_svd, recommend_user_svd


def eval_svd(
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
    n_components: int = 50,
) -> dict[str, float]:
    model = fit_svd(train, n_components=n_components)

    train_groups = {uid: g for uid, g in train.groupby("userId", sort=False)}
    test_sets = test.groupby("userId")["movieId"].apply(lambda s: set(map(int, s))).to_dict()

    precisions, recalls, ndcgs = [], [], []
    all_recommended = set()

    for uid, rel_set in test_sets.items():
        user_train = train_groups.get(uid)
        if user_train is None or not rel_set:
            continue

        recs = recommend_user_svd(model, user_train, k=k)

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