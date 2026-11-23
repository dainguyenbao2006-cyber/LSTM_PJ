"""Chuẩn bị dữ liệu: đọc 2 split, làm sạch, loại trùng, ghi `data/processed/reviews.parquet`.

Theo đặc tả trong docs/DATA.md §3-§4 và docs/phases/DATA_PREP.md:
- Dùng lại split Lab 3 (data/splits/train_val_split.json): 22.499 train / 2.501 val / 25.000 test.
- Làm sạch `<br />` và chuẩn hóa khoảng trắng với clean_text.
- Bỏ bản trùng giữa train/val và test khỏi train/val (123 review: 106 train + 17 val).
- Bỏ bản trùng lặp nội bộ trong train/val (giữ 1 bản đầu tiên: 87 train + 11 val = 98 bản).
- Giữ nguyên test (25.000 review).
- Kết quả: train = 22.306, val = 2.473, test = 25.000.
"""
from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

from src.data.clean import clean_text, text_hash
from src.data.split import apply_split, read_split
from src.utils.paths import data_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_dataset(
    raw_dir: Path | None = None,
    split_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Xử lý và tạo tập dữ liệu reviews.parquet hoàn chỉnh."""
    raw_path = raw_dir or data_dir("raw")
    out_file = output_path or (data_dir("processed") / "reviews.parquet")

    train_raw_file = raw_path / "imdb_train.parquet"
    test_raw_file = raw_path / "imdb_test.parquet"

    if not train_raw_file.exists() or not test_raw_file.exists():
        raise FileNotFoundError(
            f"Thiếu file thô trong {raw_path}. Chạy 'python -m src.data.download' trước."
        )

    logger.info("Đọc dữ liệu thô...")
    raw_train = pd.read_parquet(train_raw_file)
    raw_test = pd.read_parquet(test_raw_file)

    logger.info(f"Dữ liệu gốc: train={len(raw_train)}, test={len(raw_test)}")

    # 1. Tách train / val theo split Lab 3
    split_info = read_split(split_path)
    train_df, val_df = apply_split(raw_train, split_info)
    test_df = raw_test.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    logger.info(f"Sau split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # 2. Làm sạch văn bản và tính hash
    for df in [train_df, val_df, test_df]:
        df["text_clean"] = df["text"].map(clean_text)
        df["hash"] = df["text_clean"].map(text_hash)

    # 3. Loại review trong train/val trùng với test (tránh rò rỉ vào tập kiểm tra)
    test_hashes = set(test_df["hash"])
    train_test_overlap = train_df["hash"].isin(test_hashes)
    val_test_overlap = val_df["hash"].isin(test_hashes)

    logger.info(
        f"Review trùng giữa test và train: {train_test_overlap.sum()}, giữa test và val: {val_test_overlap.sum()}"
    )

    train_df = train_df[~train_test_overlap]
    val_df = val_df[~val_test_overlap]

    # 4. Loại review trùng nội bộ trong train và val (giữ bản ghi đầu tiên)
    train_dup = train_df.duplicated(subset=["hash"], keep="first")
    val_dup = val_df.duplicated(subset=["hash"], keep="first")

    logger.info(f"Bản trùng nội bộ train đã loại: {train_dup.sum()}, val đã loại: {val_dup.sum()}")

    train_df = train_df[~train_dup]
    val_df = val_df[~val_dup]

    logger.info(f"Số lượng sau loại trùng: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # 5. Gộp và chọn cột bàn giao: uid, split, text, text_clean, label, rating, movie_id
    columns = ["uid", "split", "text", "text_clean", "label", "rating", "movie_id"]
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)[columns]

    out_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_file, index=False)
    logger.info(f"Đã lưu kết quả vào {out_file} (tổng cộng {len(combined)} dòng)")

    return combined


if __name__ == "__main__":
    prepare_dataset()
