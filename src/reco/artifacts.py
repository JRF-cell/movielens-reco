from __future__ import annotations

import json
from pathlib import Path

from scipy import sparse

from .preprocess import RatingsData


def save_artifacts(rd: RatingsData, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save sparse matrix
    mat_path = out_dir / "user_item_ratings_csr.npz"
    sparse.save_npz(mat_path, rd.matrix)

    # Save mappings (original ids)
    meta = {
        "user_ids": rd.user_index.to_list(),
        "movie_ids": rd.item_index.to_list(),
    }
    meta_path = out_dir / "mappings.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    return out_dir
