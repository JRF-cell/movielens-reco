import numpy as np

from reco.eval import precision_recall_ndcg_at_k


def test_metrics_all_hit():
    recs = [1, 2, 3, 4, 5]
    rel = {1, 2, 3, 4, 5}
    p, r, n = precision_recall_ndcg_at_k(recs, rel, k=5)
    assert p == 1.0
    assert r == 1.0
    assert n == 1.0


def test_metrics_partial_hit():
    recs = [10, 11, 12, 13, 14]
    rel = {10, 12}
    p, r, n = precision_recall_ndcg_at_k(recs, rel, k=5)
    assert np.isclose(p, 2 / 5)
    assert np.isclose(r, 1.0)
    assert 0.0 <= n <= 1.0