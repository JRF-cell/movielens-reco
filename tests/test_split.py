import pandas as pd

from reco.eval import temporal_user_split


def test_temporal_split_keeps_last_as_test():
    df = pd.DataFrame(
        {
            "userId": [1] * 10,
            "movieId": list(range(10)),
            "rating": [4.0] * 10,
            "timestamp": list(range(10)),
        }
    )
    split = temporal_user_split(df, test_ratio=0.2, min_user_ratings=10)
    # last 2 should be in test (timestamps 8,9)
    assert set(split.test["movieId"].tolist()) == {8, 9}
    assert set(split.train["movieId"].tolist()) == set(range(8))