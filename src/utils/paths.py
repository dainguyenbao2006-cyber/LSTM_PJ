"""Đường dẫn tính từ gốc repo, chạy ở thư mục nào cũng đúng (không dùng '../').

Lấy từ Lab-DL `shared/paths.py` (nhánh feat/lab3-redo), đổi theo cấu trúc thư mục của repo này
(README §2, data/README.md, experiments/README.md).
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_KINDS = ("raw", "splits", "glove", "processed", "sample")
EXPERIMENT_KINDS = ("checkpoints", "runs", "results", "plots")


def data_dir(kind: str) -> Path:
    """Trả về (và tạo) data/<kind>: raw, splits, glove, processed hoặc sample."""
    if kind not in DATA_KINDS:
        raise ValueError(f"kind must be one of {DATA_KINDS}, got {kind!r}")
    path = REPO_ROOT / "data" / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def experiment_dir(kind: str, *parts: str) -> Path:
    """Trả về (và tạo) experiments/<kind>/<parts...>: checkpoints, runs, results hoặc plots."""
    if kind not in EXPERIMENT_KINDS:
        raise ValueError(f"kind must be one of {EXPERIMENT_KINDS}, got {kind!r}")
    path = REPO_ROOT / "experiments" / kind
    for p in parts:
        path = path / p
    path.mkdir(parents=True, exist_ok=True)
    return path
