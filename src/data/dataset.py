"""PyTorch Dataset & DataLoader cho dữ liệu phân loại cảm xúc IMDb.

Yêu cầu (docs/TASKS.md §3 D1 & docs/phases/MODEL.md):
- Trả về tensor input_ids, label và độ dài thật (lengths) để dùng với pack_padded_sequence.
- Hỗ trợ cắt ngắn (truncation) và đệm (padding) với padding_idx=0.
- Theo dõi thông tin phụ vụ phân tích lỗi: was_truncated, n_tokens, n_unk.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.tokenizer import tokenize
from src.data.vocab import PAD_IDX, UNK_IDX, Vocab


class IMDbDataset(Dataset):
    """Dataset quản lý dữ liệu review IMDb."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int] | None = None,
        uids: Sequence[str] | None = None,
        ratings: Sequence[int] | None = None,
        vocab: Vocab | None = None,
        max_len: int = 250,
        tokenizer: Callable[[str], list[str]] = tokenize,
        truncation: str = "post",
    ) -> None:
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.uids = list(uids) if uids is not None else [f"sample_{i}" for i in range(len(self.texts))]
        self.ratings = list(ratings) if ratings is not None else [0] * len(self.texts)
        self.vocab = vocab
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.truncation = truncation

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        text = self.texts[idx]
        tokens = self.tokenizer(text)
        n_tokens = len(tokens)

        # Cắt ngắn nếu vượt quá max_len
        was_truncated = n_tokens > self.max_len
        if was_truncated:
            if self.truncation == "post":
                tokens = tokens[: self.max_len]
            elif self.truncation == "pre":
                tokens = tokens[-self.max_len :]
            else:
                tokens = tokens[: self.max_len]

        if self.vocab is not None:
            input_ids = self.vocab.encode(tokens)
            n_unk = sum(1 for tid in input_ids if tid == self.vocab.unk_idx)
        else:
            input_ids = []
            n_unk = 0

        # Đảm bảo độ dài thật tối thiểu là 1 để pack_padded_sequence không bị lỗi
        length = max(1, len(input_ids))
        if len(input_ids) == 0:
            input_ids = [self.vocab.unk_idx if self.vocab else UNK_IDX]

        item: dict[str, Any] = {
            "uid": self.uids[idx],
            "text": text,
            "input_ids": input_ids,
            "length": length,
            "n_tokens": n_tokens,
            "n_unk": n_unk,
            "was_truncated": was_truncated,
            "rating": self.ratings[idx],
        }

        if self.labels is not None:
            item["label"] = float(self.labels[idx])

        return item


def collate_imdb_batch(
    batch: list[dict[str, Any]],
    pad_idx: int = PAD_IDX,
    fixed_len: int | None = None,
) -> dict[str, Any]:
    """Hàm gộp batch, đệm động đến độ dài lớn nhất trong batch (hoặc fixed_len)."""
    batch_size = len(batch)
    lengths = [item["length"] for item in batch]
    max_len = fixed_len if fixed_len is not None else max(lengths)

    # Khởi tạo tensor đệm sẵn
    padded_inputs = torch.full((batch_size, max_len), fill_value=pad_idx, dtype=torch.long)
    for i, item in enumerate(batch):
        seq = item["input_ids"][:max_len]
        padded_inputs[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    res: dict[str, Any] = {
        "input_ids": padded_inputs,
        "lengths": lengths_tensor,
        "uids": [item["uid"] for item in batch],
        "texts": [item["text"] for item in batch],
        "ratings": [item["rating"] for item in batch],
        "n_tokens": [item["n_tokens"] for item in batch],
        "n_unk": [item["n_unk"] for item in batch],
        "was_truncated": [item["was_truncated"] for item in batch],
    }

    if "label" in batch[0]:
        res["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

    return res


def create_dataloader(
    dataset: IMDbDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pad_idx: int = PAD_IDX,
    fixed_len: int | None = None,
) -> DataLoader:
    """Tạo DataLoader tương thích đa luồng và Windows."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_imdb_batch(b, pad_idx=pad_idx, fixed_len=fixed_len),
    )


def load_imdb_split_data(
    parquet_path: str | None = None,
    split: str = "train",
) -> pd.DataFrame:
    """Đọc dữ liệu theo split từ file processed parquet."""
    from src.utils.paths import data_dir

    p = parquet_path or (data_dir("processed") / "reviews.parquet")
    df = pd.read_parquet(p)
    return df[df["split"] == split].reset_index(drop=True)
