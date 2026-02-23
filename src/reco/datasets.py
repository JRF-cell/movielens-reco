from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    url: str
    folder: str


MOVIELENS: dict[str, DatasetSpec] = {
    "latest-small": DatasetSpec(
        key="latest-small",
        url="https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        folder="ml-latest-small",
    ),
    "latest": DatasetSpec(
        key="latest",
        url="https://files.grouplens.org/datasets/movielens/ml-latest.zip",
        folder="ml-latest",
    ),
}


def _download_to(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, dst.open("wb") as f:
        shutil.copyfileobj(r, f)


def _safe_extract(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_root = dest_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            # Prevent Zip Slip
            out_path = (dest_dir / member.filename).resolve()
            if not str(out_path).startswith(str(dest_root)):
                raise ValueError(f"Unsafe path in zip: {member.filename}")
        zf.extractall(dest_dir)


def download_movielens(dataset: str = "latest-small", data_dir: str | Path = "data", force: bool = False) -> Path:
    if dataset not in MOVIELENS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {', '.join(MOVIELENS)}")

    spec = MOVIELENS[dataset]
    data_dir = Path(data_dir)
    target_dir = data_dir / spec.folder

    if target_dir.exists():
        if not force:
            return target_dir
        shutil.rmtree(target_dir)

    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td) / f"{spec.folder}.zip"
        _download_to(spec.url, zip_path)
        _safe_extract(zip_path, data_dir)

    ratings = target_dir / "ratings.csv"
    movies = target_dir / "movies.csv"
    if not (ratings.exists() and movies.exists()):
        raise FileNotFoundError(f"Dataset extracted but expected files missing in {target_dir}")

    return target_dir
