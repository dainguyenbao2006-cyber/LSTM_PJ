"""Unit tests cho Baseline M0a (Majority) và M0b (TF-IDF + LR)."""
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.models.baseline import BaselineRunner


def test_baseline_runner_mock():
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / "reviews.parquet"
        df = pd.DataFrame({
            "uid": [f"u_{i}" for i in range(20)],
            "split": ["train"] * 10 + ["val"] * 5 + ["test"] * 5,
            "text": ["good movie", "bad film"] * 10,
            "text_clean": ["good movie", "bad film"] * 10,
            "label": [1, 0] * 10,
            "rating": [8, 2] * 10,
            "movie_id": [f"m_{i}" for i in range(20)],
        })
        df.to_parquet(p)

        runner = BaselineRunner(data_path=str(p))
        m0a = runner.run_majority_baseline()
        assert "majority_label" in m0a
        assert "val_metrics" in m0a
        assert "test_metrics" in m0a

        m0b = runner.run_tfidf_lr_baseline(c_candidates=[1.0], save_artifacts=False)
        assert "best_c" in m0b
        assert m0b["best_c"] == 1.0
        assert "val_metrics" in m0b
        assert "test_metrics" in m0b
