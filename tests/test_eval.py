"""Test cho Task D8: Metrics, McNemar test, và quy trình đánh giá."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.eval.metrics import (
    compute_classification_metrics,
    mcnemar_test,
    measure_inference_speed,
)
from src.evaluate import evaluate_split, compare_all_models
from src.models.rnn import SentimentRNN
from src.data.vocab import Vocab


def test_compute_classification_metrics():
    y_true = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0]
    y_prob = [0.9, 0.1, 0.8, 0.4, 0.2, 0.7, 0.85, 0.3]

    res = compute_classification_metrics(y_true, y_pred, y_prob)

    assert res["accuracy"] == 6 / 8
    assert "macro_f1" in res
    assert res["confusion_matrix"]["tp"] == 3
    assert res["confusion_matrix"]["tn"] == 3
    assert res["confusion_matrix"]["fp"] == 1
    assert res["confusion_matrix"]["fn"] == 1
    assert res["confusion_matrix"]["total"] == 8
    assert res["roc_auc"] is not None


def test_mcnemar_test():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    # Model A: 7 correct, 1 wrong
    y_pred_a = np.array([1, 1, 1, 1, 0, 0, 0, 1])
    # Model B: 4 correct, 4 wrong
    y_pred_b = np.array([1, 0, 0, 1, 1, 1, 0, 0])

    res = mcnemar_test(y_true, y_pred_a, y_pred_b, "Model A", "Model B")

    assert res["model_a"] == "Model A"
    assert res["model_b"] == "Model B"
    assert "statistic" in res
    assert "p_value" in res
    assert res["contingency_table"]["total"] == 8
    # b: A đúng, B sai; c: A sai, B đúng
    assert res["contingency_table"]["a_correct_b_wrong"] >= 0


def test_evaluate_split_pipeline():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Tạo mock vocab
        vocab = Vocab({"<pad>": 0, "<unk>": 1, "good": 2, "bad": 3})
        vocab_file = tmp_path / "vocab.json"
        vocab.save(vocab_file)

        # 2. Tạo mock checkpoint
        model = SentimentRNN(vocab_size=len(vocab), embed_dim=8, hidden_dim=16, rnn_type="lstm")
        ckpt_file = tmp_path / "best.pt"
        model.save_checkpoint(str(ckpt_file), config={"data": {"max_len": 10}})

        # 3. Tạo mock parquet data
        df = pd.DataFrame({
            "uid": ["test_0", "test_1", "test_2", "test_3"],
            "split": ["test", "test", "test", "test"],
            "text": ["good film", "bad film", "very good", "so bad"],
            "text_clean": ["good film", "bad film", "very good", "so bad"],
            "label": [1, 0, 1, 0],
            "rating": [9, 2, 8, 1],
            "movie_id": ["m1", "m2", "m3", "m4"],
        })
        parquet_file = tmp_path / "reviews.parquet"
        df.to_parquet(parquet_file)

        # 4. Chạy evaluate_split
        out_dir = tmp_path / "results"
        results = evaluate_split(
            checkpoint_path=ckpt_file,
            vocab_path=vocab_file,
            split="test",
            data_path=str(parquet_file),
            output_dir=out_dir,
            device_name="cpu",
            batch_size=2,
        )

        assert results["split"] == "test"
        assert (out_dir / "predictions_test.csv").exists()
        assert (out_dir / "test_metrics.json").exists()

        pred_df = pd.read_csv(out_dir / "predictions_test.csv")
        assert len(pred_df) == 4
        # Kiểm tra đầy đủ các cột bàn giao bắt buộc
        expected_cols = {"uid", "text", "label", "rating", "prob", "pred", "n_tokens", "n_unk", "was_truncated"}
        assert expected_cols.issubset(set(pred_df.columns))
