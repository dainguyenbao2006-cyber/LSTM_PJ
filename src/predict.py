"""Dự đoán cảm xúc cho một hoặc nhiều câu văn bản (Inference API / CLI).

Bàn giao Đại → Bảo (docs/TASKS.md §4):
- Hàm `predict(text: str) -> tuple[int, float]` trả về (nhãn, xác suất tích cực).
- CLI để kiểm tra nhanh một câu từ terminal.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from src.data.dataset import collate_imdb_batch
from src.data.tokenizer import tokenize
from src.data.vocab import Vocab
from src.models.rnn import SentimentRNN


class SentimentPredictor:
    """Class tiện ích tải checkpoint và thực hiện suy luận nhanh."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        vocab_path: str | Path,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.vocab = Vocab.load(vocab_path)
        self.model, self.ckpt = SentimentRNN.load_checkpoint(str(checkpoint_path), device=self.device)
        self.model.eval()

        self.max_len = self.ckpt.get("config", {}).get("data", {}).get("max_len", 250)

    def predict(self, text: str) -> tuple[int, float]:
        """Dự đoán cho một chuỗi văn bản.

        Returns:
            label: 1 (Positive) hoặc 0 (Negative)
            prob: xác suất của lớp Positive (0.0 đến 1.0)
        """
        probs = self.predict_batch([text])
        prob = probs[0]
        label = 1 if prob >= 0.5 else 0
        return label, prob

    def predict_batch(self, texts: list[str]) -> list[float]:
        """Dự đoán xác suất positive cho một danh sách chuỗi."""
        items = []
        for uid_idx, t in enumerate(texts):
            tokens = tokenize(t)
            tokens = tokens[: self.max_len]
            input_ids = self.vocab.encode(tokens)
            length = max(1, len(input_ids))
            if not input_ids:
                input_ids = [self.vocab.unk_idx]
            items.append({
                "uid": f"pred_{uid_idx}",
                "text": t,
                "input_ids": input_ids,
                "length": length,
                "rating": 0,
                "n_tokens": len(tokens),
                "n_unk": 0,
                "was_truncated": False,
            })

        batch = collate_imdb_batch(items, pad_idx=self.vocab.pad_idx)
        input_ids = batch["input_ids"].to(self.device)
        lengths = batch["lengths"]

        with torch.no_grad():
            probs = self.model.predict_proba(input_ids, lengths).cpu().tolist()

        if isinstance(probs, float):
            return [probs]
        return probs


def get_default_predictor() -> SentimentPredictor:
    """Tải checkpoint mặc định có sẵn trong experiments/checkpoints."""
    ckpt_root = Path("experiments/checkpoints")
    pts = list(ckpt_root.glob("*/seed_*/best.pt"))
    if not pts:
        raise FileNotFoundError("Chưa tìm thấy checkpoint 'best.pt' nào trong experiments/checkpoints/")

    ckpt_path = pts[0]
    exp_dir = ckpt_path.parents[1]
    vocab_path = exp_dir / "vocab.json"
    if not vocab_path.exists():
        # Fallback to processed vocab
        vocab_path = Path("data/processed/vocab.json")

    return SentimentPredictor(checkpoint_path=ckpt_path, vocab_path=vocab_path)


