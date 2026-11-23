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
) -> tuple[np.ndarray, float]:
    """Trích xuất ma trận embedding từ file GloVe theo từ vựng của tập train.

    Returns:
        matrix: ma trận numpy shape [len(vocab), embed_dim]
        coverage: tỷ lệ từ trong vocab có mặt trong GloVe (không tính <pad>)
    """
    rng = np.random.RandomState(seed)
    matrix = rng.normal(loc=0.0, scale=0.1, size=(len(vocab), embed_dim)).astype(np.float32)

    # <pad> luôn đặt bằng 0
    matrix[vocab.pad_idx] = np.zeros(embed_dim, dtype=np.float32)

    glove_file = Path(glove_path) if glove_path else (data_dir("glove") / f"glove.6B.{embed_dim}d.txt")

    found_count = 0
    if glove_file.exists():
        logger.info(f"Đang đọc GloVe từ {glove_file}...")
        with glove_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ")
                word = parts[0]
                if word in vocab and word != vocab.pad_token:
                    idx = vocab[word]
                    try:
                        vec = np.asarray(parts[1:], dtype=np.float32)
                        if len(vec) == embed_dim:
                            matrix[idx] = vec
                            found_count += 1
                    except ValueError:
                        continue
    else:
        logger.warning(
            f"Không tìm thấy file GloVe tại {glove_file}. Khởi tạo ngẫu nhiên toàn bộ trừ <pad>."
        )

    # Tỷ lệ phủ không tính token <pad>
    total_words = max(1, len(vocab) - 1)
    coverage = float(found_count / total_words)
    logger.info(
        f"GloVe coverage: {found_count}/{total_words} từ ({coverage * 100:.2f}%)"
    )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, matrix)
        logger.info(f"Đã lưu ma trận GloVe vào {out}")

    return matrix, coverage


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Trích xuất GloVe theo từ vựng.")
    parser.add_argument("--vocab", type=str, default=None, help="Đường dẫn vocab.json")
    parser.add_argument("--glove", type=str, default=None, help="Đường dẫn glove txt")
    parser.add_argument("--dim", type=int, default=100, help="Số chiều GloVe (mặc định 100)")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn lưu .npy")
    args = parser.parse_args()

    vocab_file = Path(args.vocab) if args.vocab else (data_dir("processed") / "vocab.json")
    if not vocab_file.exists():
        raise FileNotFoundError(f"Không tìm thấy vocab tại {vocab_file}")

    v = Vocab.load(vocab_file)
    out_file = Path(args.output) if args.output else (data_dir("processed") / f"glove_{args.dim}d_vocab.npy")
    build_glove_matrix(v, glove_path=args.glove, embed_dim=args.dim, output_path=out_file)
