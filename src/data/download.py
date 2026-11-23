"""Tải và đọc IMDb gốc (aclImdb v1, Maas et al. 2011): nhãn, rating, mã phim của từng review.

Lấy từ Lab-DL `labs/lab3/preprocessing/preprocessing.py` (nhánh feat/lab3-redo), giữ nguyên logic.
Dùng bản gốc Stanford thay vì `stanfordnlp/imdb` vì chỉ bản gốc có mã phim (`urls_*.txt`) và rating.

Chạy từ gốc repo:
    python -m src.data.download      # tải (nếu chưa có), đọc 2 split, in số dòng
"""
from __future__ import annotations

import re
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

from src.utils.paths import data_dir

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
SPLITS = ("train", "test")
LABEL_DIRS = {"neg": 0, "pos": 1}  # thư mục neg -> 0, pos -> 1
REVIEWS_PER_DIR = 12_500  # README aclImdb: 25k train + 25k test, cân bằng nhãn
_MOVIE_RE = re.compile(r"/title/(tt\d+)/")


def raw_root(root: Path | None = None) -> Path:
    """Thư mục chứa aclImdb/ và bộ đệm parquet (mặc định data/raw, git bỏ qua)."""
    return Path(root) if root is not None else data_dir("raw")


def _cache_path(split: str, root: Path | None = None) -> Path:
    return raw_root(root) / f"imdb_{split}.parquet"


def download_imdb(root: Path | None = None, url: str = IMDB_URL) -> Path:
    """Tải và giải nén aclImdb_v1.tar.gz một lần; kiểm 4 thư mục × 12.500 file.

    Nếu đã có bộ đệm parquet của cả 2 split thì bỏ qua (đọc parquet nhanh hơn nhiều).
    """
    root = raw_root(root)
    imdb = root / "aclImdb"
    if all(_cache_path(s, root).exists() for s in SPLITS):
        return imdb
    if not (imdb / "test" / "pos").is_dir():
        archive = root / "aclImdb_v1.tar.gz"
        if not archive.exists():
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(root, filter="data")  # filter="data": chặn đường dẫn lạ trong file nén
    for split in SPLITS:
        for label_dir in LABEL_DIRS:
            n = len(list((imdb / split / label_dir).glob("*.txt")))
            if n != REVIEWS_PER_DIR:
                raise ValueError(f"{split}/{label_dir}: expected {REVIEWS_PER_DIR} files, found {n}")
    return imdb


def read_movie_ids(urls_file: Path) -> list[str]:
    """Mã phim (tt...) của từng dòng urls_[pos|neg].txt; README aclImdb: review id i ở dòng i.

    Dòng không có mã tt... thì báo lỗi (không đoán).
    """
    ids = []
    for i, line in enumerate(Path(urls_file).read_text(encoding="utf-8").splitlines()):
        m = _MOVIE_RE.search(line)
        if m is None:
            raise ValueError(f"{urls_file} line {i}: no movie id in {line!r}")
        ids.append(m.group(1))
    return ids


def load_split(split: str, root: Path | None = None) -> pd.DataFrame:
    """Đọc một split chính thức (train hoặc test) thành bảng.

    Cột: uid, split, id, rating, label, movie_id, text (thô, chưa làm sạch).
    Tên file <id>_<rating>.txt; nhãn theo thư mục. Số dòng URL phải bằng số review, nếu không thì dừng.
    Bộ đệm: đọc 25k file nhỏ trên Windows rất chậm (đo ở Lab 3, 2026-11-20) → lần đầu đọc xong thì lưu
    data/raw/imdb_<split>.parquet; lần sau đọc file này. Xoá file để đọc lại từ gốc.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    cache = _cache_path(split, root)
    if cache.exists():
        return pd.read_parquet(cache)
    imdb = raw_root(root) / "aclImdb"
    rows = []
    for label_dir, label in LABEL_DIRS.items():
        files = sorted((imdb / split / label_dir).glob("*.txt"))
        movies = read_movie_ids(imdb / split / f"urls_{label_dir}.txt")
        if len(movies) != len(files):
            raise ValueError(f"{split}/{label_dir}: {len(files)} reviews but {len(movies)} URL lines")
        for f in files:
            rid, rating = (int(x) for x in f.stem.split("_"))
            rows.append({
                "uid": f"{split}_{label_dir}_{rid}",
                "split": split,
                "id": rid,
                "rating": rating,
                "label": label,
                "movie_id": movies[rid],
                "text": f.read_text(encoding="utf-8"),
            })
    df = pd.DataFrame(rows).sort_values("uid", kind="stable").reset_index(drop=True)
    df.to_parquet(cache, index=False)
    return df


def main() -> None:
    download_imdb()
    for split in SPLITS:
        df = load_split(split)
        print(f"{split}: {len(df)} reviews, {df['label'].sum()} positive, {df['movie_id'].nunique()} movies")


if __name__ == "__main__":
    main()
