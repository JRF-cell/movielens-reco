# src/reco/cli.py
import argparse

import pandas as pd

from .artifacts import save_artifacts
from .datasets import MOVIELENS, download_movielens
from .eval import eval_popularity, temporal_user_split
from .eval_itemcos import eval_item_cosine
from .eval_svd import eval_svd
from .preprocess import basic_stats, build_user_item_matrix, load_ratings_csv
from .recommend import load_movies, recommend_for_user
from .svd import fit_svd, recommend_user_svd


def cmd_health(_: argparse.Namespace) -> int:
    print("ok")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    out = download_movielens(dataset=args.dataset, data_dir=args.data_dir, force=args.force)
    print(str(out))
    return 0


def _dataset_dir(data_dir: str, dataset: str) -> str:
    if dataset == "latest":
        return f"{data_dir}/ml-latest"
    return f"{data_dir}/ml-latest-small"


def cmd_preprocess(args: argparse.Namespace) -> int:
    dataset_dir = _dataset_dir(args.data_dir, args.dataset)
    ratings = load_ratings_csv(dataset_dir)
    rd = build_user_item_matrix(ratings)
    stats = basic_stats(rd)

    if args.out:
        save_artifacts(rd, args.out)

    for k, v in stats.items():
        print(f"{k}={v}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    dataset_dir = _dataset_dir(args.data_dir, args.dataset)
    ratings = load_ratings_csv(dataset_dir)
    split = temporal_user_split(
        ratings,
        test_ratio=args.test_ratio,
        min_user_ratings=args.min_user_ratings,
    )
    res = eval_popularity(split.train, split.test, k=args.k)
    for k, v in res.items():
        print(f"{k}={v}")
    return 0


def cmd_evaluate_itemcos(args: argparse.Namespace) -> int:
    dataset_dir = _dataset_dir(args.data_dir, args.dataset)
    ratings = load_ratings_csv(dataset_dir)
    split = temporal_user_split(
        ratings,
        test_ratio=args.test_ratio,
        min_user_ratings=args.min_user_ratings,
    )
    res = eval_item_cosine(
        split.train,
        split.test,
        k=args.k,
        n_seed=args.n_seed,
        n_neighbors=args.n_neighbors,
    )
    for k, v in res.items():
        print(f"{k}={v}")
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    dataset_dir = _dataset_dir(args.data_dir, args.dataset)
    ratings = load_ratings_csv(dataset_dir)
    movies = load_movies(dataset_dir)

    out = recommend_for_user(
        ratings=ratings,
        movies=movies,
        user_id=args.user_id,
        k=args.k,
        test_ratio=args.test_ratio,
        min_user_ratings=args.min_user_ratings,
        n_seed=args.n_seed,
        n_neighbors=args.n_neighbors,
    )

    for i, row in enumerate(out.itertuples(index=False), start=1):
        title = row.title if isinstance(row.title, str) else "<unknown title>"
        flag = " (in test)" if row.in_user_test else ""
        print(f"{i:02d}. {title}{flag}")
    return 0


def cmd_evaluate_svd(args: argparse.Namespace) -> int:
    dataset_dir = _dataset_dir(args.data_dir, args.dataset)
    ratings = load_ratings_csv(dataset_dir)
    split = temporal_user_split(
        ratings,
        test_ratio=args.test_ratio,
        min_user_ratings=args.min_user_ratings,
    )
    res = eval_svd(
        split.train,
        split.test,
        k=args.k,
        n_components=args.n_components,
    )
    for k, v in res.items():
        print(f"{k}={v}")
    return 0


def cmd_recommend_svd(args: argparse.Namespace) -> int:
    dataset_dir = _dataset_dir(args.data_dir, args.dataset)
    ratings = load_ratings_csv(dataset_dir)
    movies = load_movies(dataset_dir)

    split = temporal_user_split(
        ratings,
        test_ratio=args.test_ratio,
        min_user_ratings=args.min_user_ratings,
    )
    train, test = split.train, split.test

    if args.user_id not in set(train["userId"].unique()):
        raise ValueError(f"userId={args.user_id} not found in train split (or too few ratings).")

    model = fit_svd(train, n_components=args.n_components)
    user_train = train[train["userId"] == args.user_id]
    rec_ids = recommend_user_svd(model, user_train, k=args.k)

    out = pd.DataFrame({"movieId": rec_ids}).merge(movies[["movieId", "title"]], on="movieId", how="left")
    test_set = set(test[test["userId"] == args.user_id]["movieId"].astype(int).tolist())
    out["in_user_test"] = out["movieId"].apply(lambda x: int(x) in test_set)

    for i, row in enumerate(out.itertuples(index=False), start=1):
        title = row.title if isinstance(row.title, str) else "<unknown title>"
        flag = " (in test)" if row.in_user_test else ""
        print(f"{i:02d}. {title}{flag}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reco", description="MovieLens recommender CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    health = sub.add_parser("health", help="Sanity check")
    health.set_defaults(func=cmd_health)

    dl = sub.add_parser("download", help="Download and extract MovieLens dataset into data/")
    dl.add_argument("--dataset", choices=sorted(MOVIELENS.keys()), default="latest-small")
    dl.add_argument("--data-dir", default="data", type=str)
    dl.add_argument("--force", action="store_true", help="Re-download and overwrite existing dataset folder")
    dl.set_defaults(func=cmd_download)

    pp = sub.add_parser("preprocess", help="Load ratings and build sparse user-item matrix (stats only)")
    pp.add_argument("--dataset", choices=sorted(MOVIELENS.keys()), default="latest-small")
    pp.add_argument("--data-dir", default="data", type=str)
    pp.add_argument("--out", default="artifacts", type=str, help="Where to save matrix + mappings")
    pp.set_defaults(func=cmd_preprocess)

    ev = sub.add_parser("evaluate", help="Temporal split + popularity baseline evaluation")
    ev.add_argument("--dataset", choices=sorted(MOVIELENS.keys()), default="latest-small")
    ev.add_argument("--data-dir", default="data", type=str)
    ev.add_argument("--k", default=10, type=int)
    ev.add_argument("--test-ratio", default=0.2, type=float)
    ev.add_argument("--min-user-ratings", default=10, type=int)
    ev.set_defaults(func=cmd_evaluate)

    ev2 = sub.add_parser("evaluate-itemcos", help="Temporal split + item-item cosine evaluation")
    ev2.add_argument("--dataset", choices=sorted(MOVIELENS.keys()), default="latest-small")
    ev2.add_argument("--data-dir", default="data", type=str)
    ev2.add_argument("--k", default=10, type=int)
    ev2.add_argument("--test-ratio", default=0.2, type=float)
    ev2.add_argument("--min-user-ratings", default=10, type=int)
    ev2.add_argument("--n-seed", default=20, type=int, help="How many recent items per user to use as seeds")
    ev2.add_argument("--n-neighbors", default=50, type=int, help="Neighbors per seed item")
    ev2.set_defaults(func=cmd_evaluate_itemcos)

    rec = sub.add_parser("recommend", help="Recommend top-K movies for a given userId (item-item cosine)")
    rec.add_argument("user_id", type=int)
    rec.add_argument("--dataset", choices=sorted(MOVIELENS.keys()), default="latest-small")
    rec.add_argument("--data-dir", default="data", type=str)
    rec.add_argument("--k", default=10, type=int)
    rec.add_argument("--test-ratio", default=0.2, type=float)
    rec.add_argument("--min-user-ratings", default=10, type=int)
    rec.add_argument("--n-seed", default=20, type=int)
    rec.add_argument("--n-neighbors", default=50, type=int)
    rec.set_defaults(func=cmd_recommend)

    ev3 = sub.add_parser("evaluate-svd", help="Temporal split + SVD (matrix factorization) evaluation")
    ev3.add_argument("--dataset", choices=sorted(MOVIELENS.keys()), default="latest-small")
    ev3.add_argument("--data-dir", default="data", type=str)
    ev3.add_argument("--k", default=10, type=int)
    ev3.add_argument("--test-ratio", default=0.2, type=float)
    ev3.add_argument("--min-user-ratings", default=10, type=int)
    ev3.add_argument("--n-components", default=50, type=int)
    ev3.set_defaults(func=cmd_evaluate_svd)

    rec2 = sub.add_parser("recommend-svd", help="Recommend top-K movies for a given userId (SVD)")
    rec2.add_argument("user_id", type=int)
    rec2.add_argument("--dataset", choices=sorted(MOVIELENS.keys()), default="latest-small")
    rec2.add_argument("--data-dir", default="data", type=str)
    rec2.add_argument("--k", default=10, type=int)
    rec2.add_argument("--test-ratio", default=0.2, type=float)
    rec2.add_argument("--min-user-ratings", default=10, type=int)
    rec2.add_argument("--n-components", default=50, type=int)
    rec2.set_defaults(func=cmd_recommend_svd)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))