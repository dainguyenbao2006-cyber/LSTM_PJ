"""Chia train/val theo phim, dùng lại file split của Lab 3 (Thư chốt 2026-11-23).

File `data/splits/train_val_split.json` chép từ Lab-DL `labs/lab3/data/splits/train_val_split.json`
(nhánh feat/lab3-redo): StratifiedGroupKFold 10 fold, seed 42, lấy fold 0 làm val → 22.499 / 2.501.
Nhóm = mã phim, gộp thêm các review trùng nội dung (union-find) để cả cặp trùng nằm cùng một phía.
Lý do dùng lại: cùng train/val với DistilBERT của Lab 3 nên so trực tiếp được (Discussion/pj/plan_pj.md §3).

`make_train_val_split` giữ lại để giải thích và kiểm lại cách tạo file; dự án KHÔNG tạo split mới.
Loại review trùng (plan D3) làm SAU khi áp split này — phần việc của data (docs/phases/DATA_PREP.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.clean import text_hash
from src.utils.paths import data_dir

SPLIT_SEED = 42
VAL_FOLDS = 10


def default_split_path() -> Path:
    return data_dir("splits") / "train_val_split.json"


def split_groups(df: pd.DataFrame) -> pd.Series:
    """Mã nhóm của từng review: cùng phim HOẶC cùng nội dung đã làm sạch → cùng nhóm (union-find)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    hashes = df["text"].map(text_hash)
    first_movie_of_hash: dict[str, str] = {}
    for movie, h in zip(df["movie_id"], hashes):
        find(movie)
        if h in first_movie_of_hash:
            union(movie, first_movie_of_hash[h])
        else:
            first_movie_of_hash[h] = movie
    return df["movie_id"].map(find)


def make_train_val_split(train: pd.DataFrame, seed: int = SPLIT_SEED, n_folds: int = VAL_FOLDS) -> dict:
    """Cách Lab 3 tạo file split: lấy fold 0 của StratifiedGroupKFold làm val (~1/n_folds train).

    Cân bằng nhãn và không nhóm nào (phim / review trùng) nằm ở cả train và val.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    groups = split_groups(train)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf.split(train, train["label"], groups))
    is_val = pd.Series(False, index=train.index)
    is_val.iloc[val_idx] = True
    tr, va = train[~is_val], train[is_val]
    return {
        "method": f"StratifiedGroupKFold(n_splits={n_folds}, shuffle=True, random_state={seed}); fold 0 = val",
        "group": "movie_id, merged with duplicate texts (union-find)",
        "seed": seed,
        "n_train": len(tr), "n_val": len(va),
        "val_pos_share": round(float(va["label"].mean()), 4),
        "n_groups": int(groups.nunique()),
        "groups_in_both": int(len(set(groups[~is_val]) & set(groups[is_val]))),
        "train_uids": tr["uid"].tolist(),
        "val_uids": va["uid"].tolist(),
    }


def save_split(split: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def read_split(path: Path | None = None) -> dict:
    path = Path(path) if path is not None else default_split_path()
    return json.loads(path.read_text(encoding="utf-8"))


def apply_split(train_all: pd.DataFrame, split: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tách bảng train chính thức (25k) thành (train, val) theo danh sách uid trong file split.

    Dừng nếu file split nhắc tới uid không có trong bảng (sai dữ liệu hoặc sai file).
    """
    split = split if split is not None else read_split()
    by_uid = train_all.set_index("uid")
    missing = (set(split["train_uids"]) | set(split["val_uids"])) - set(by_uid.index)
    if missing:
        raise ValueError(f"{len(missing)} uid in split file not found in data, e.g. {sorted(missing)[:3]}")
    train = by_uid.loc[split["train_uids"]].reset_index()
    val = by_uid.loc[split["val_uids"]].reset_index()
    return train, val
