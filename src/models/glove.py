"""Xử lý và trích xuất ma trận embedding GloVe theo từ vựng (Vocabulary).

Yêu cầu (docs/TASKS.md §3 D3, docs/DATA.md §6 & docs/phases/MODEL.md):
- Nạp GloVe 6B 100d.
- Lọc theo từ vựng tập train.
- <pad> khởi tạo bằng vector 0.
- Các từ OOV / không có trong GloVe được khởi tạo ngẫu nhiên.
- Tính và ghi nhận tỷ lệ phủ (coverage).
- Lưu ma trận ra data/processed/glove_100d_vocab.npy.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import numpy as np

from src.data.vocab import PAD_IDX, Vocab
from src.utils.paths import data_dir

logger = logging.getLogger(__name__)


def build_glove_matrix(
    vocab: Vocab,
    glove_path: Path | str | None = None,
    embed_dim: int = 100,
    output_path: Path | str | None = None,
    seed: int = 42,
    allow_random: bool = False,      # chỉ True khi cố ý chạy thử không có GloVe
    min_coverage: float = 0.80,      # dưới ngưỡng này coi như lỗi
) -> tuple[np.ndarray, float]:
    rng = np.random.RandomState(seed)
    matrix = rng.normal(loc=0.0, scale=0.1, size=(len(vocab), embed_dim)).astype(np.float32)
    matrix[vocab.pad_idx] = 0.0

    glove_file = Path(glove_path) if glove_path else (data_dir("glove") / f"glove.6B.{embed_dim}d.txt")

    found_count = 0
    if glove_file.exists():
        logger.info(f"Đang đọc GloVe từ {glove_file}...")
        with glove_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                if word in vocab and word != vocab.pad_token:
                    vec = np.asarray(parts[1:], dtype=np.float32)  # ValueError nếu file hỏng -> để nổi lên
                    if len(vec) == embed_dim:
                        matrix[vocab[word]] = vec
                        found_count += 1
    elif allow_random:
        logger.warning(f"Không tìm thấy GloVe tại {glove_file}. Dùng random (allow_random=True).")
    else:
        raise FileNotFoundError(
            f"Không tìm thấy file GloVe: {glove_file}. "
            "Tải glove.6B.zip và giải nén glove.6B.100d.txt vào data/glove/."
        )

    total_words = max(1, len(vocab) - 1)          # không tính <pad>
    coverage = found_count / total_words
    logger.info(f"GloVe coverage: {found_count}/{total_words} ({coverage*100:.2f}%)")

    if not allow_random and coverage < min_coverage:
        raise ValueError(
            f"Coverage GloVe chỉ {coverage:.2%} (< {min_coverage:.0%}), "
            "kiểm tra lại file GloVe hoặc tokenizer/vocab."
        )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, matrix)
        logger.info(f"Đã lưu ma trận GloVe vào {out}")

    return matrix, float(coverage)
