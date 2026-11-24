"""Mô hình Baseline: Majority Class (M0a) và TF-IDF + Logistic Regression (M0b).

Theo đặc tả trong docs/phases/BASELINE.md:
- M0a: Majority class predictor
- M0b: TfidfVectorizer(ngram_range=(1,2), min_df=2) + LogisticRegression(C=...)
- Tìm C tối ưu trên val set
- Lưu checkpoint tại experiments/checkpoints/baseline_tfidf_lr.joblib
- Xuất predictions_<split>.csv và metrics.json cho cả val và test
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data.dataset import load_imdb_split_data
from src.data.tokenizer import tokenize
from src.eval.metrics import compute_classification_metrics
from src.utils.paths import experiment_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("baseline")


class BaselineRunner:
    def __init__(self, data_path: str | Path | None = None) -> None:
        self.data_path = data_path
        self.train_df = load_imdb_split_data(parquet_path=data_path, split="train")
        self.val_df = load_imdb_split_data(parquet_path=data_path, split="val")
        self.test_df = load_imdb_split_data(parquet_path=data_path, split="test")

    def run_majority_baseline(self) -> dict[str, Any]:
        """Chạy mô hình Majority Class (M0a)."""
        logger.info("Chạy Baseline M0a (Majority Class)...")
        majority_label = int(self.train_df["label"].mode()[0])
        logger.info(f"Nhãn đa số trong train: {majority_label}")

        results: dict[str, Any] = {"model": "M0a_Majority", "majority_label": majority_label}

        for split_name, df in [("val", self.val_df), ("test", self.test_df)]:
            targets = df["label"].values.tolist()
            preds = [majority_label] * len(df)
            probs = [1.0 if majority_label == 1 else 0.0] * len(df)
            metrics = compute_classification_metrics(targets, preds, probs)
            results[f"{split_name}_metrics"] = metrics
            logger.info(f"M0a {split_name} Acc: {metrics['accuracy'] * 100:.2f}%")

            out_dir = Path("experiments/results/majority")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{split_name}_metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        return results

    def run_tfidf_lr_baseline(
        self,
        c_candidates: list[float] | None = None,
        save_artifacts: bool = True,
    ) -> dict[str, Any]:
        """Huấn luyện và đánh giá TF-IDF + Logistic Regression (M0b)."""
        if c_candidates is None:
            c_candidates = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

        logger.info("Khởi tạo TfidfVectorizer (ngram_range=(1, 2), min_df=2)...")
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
        )

        logger.info("Fit TF-IDF trên tập train...")
        x_train = vectorizer.fit_transform(self.train_df["text_clean"])
        y_train = self.train_df["label"].values

        x_val = vectorizer.transform(self.val_df["text_clean"])
        y_val = self.val_df["label"].values

        # Tìm C tốt nhất trên val
        best_c = 1.0
        best_val_acc = -1.0
        val_results: dict[float, float] = {}

        for c_val in c_candidates:
            clf = LogisticRegression(C=c_val, max_iter=1000, solver="lbfgs", random_state=42)
            clf.fit(x_train, y_train)
            val_preds = clf.predict(x_val)
            acc = float((val_preds == y_val).mean())
            val_results[c_val] = acc
            logger.info(f"C={c_val:<5} -> Val Acc: {acc * 100:.2f}%")
            if acc > best_val_acc:
                best_val_acc = acc
                best_c = c_val

        logger.info(f"Chọn C tối ưu: {best_c} với Val Acc: {best_val_acc * 100:.2f}%")

        # Huấn luyện mô hình cuối với C tốt nhất
        best_clf = LogisticRegression(C=best_c, max_iter=1000, solver="lbfgs", random_state=42)
        best_clf.fit(x_train, y_train)

        x_test = vectorizer.transform(self.test_df["text_clean"])
        y_test = self.test_df["label"].values

        results: dict[str, Any] = {
            "model": "M0b_TFIDF_LR",
            "best_c": best_c,
            "c_tuning": val_results,
        }

        # Lưu checkpoint
        ckpt_dir = Path("experiments/checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "baseline_tfidf_lr.joblib"
        joblib.dump({"vectorizer": vectorizer, "classifier": best_clf, "best_c": best_c}, ckpt_path)
        logger.info(f"Đã lưu mô hình M0b vào {ckpt_path}")

        # Đánh giá và xuất kết quả
        out_res_dir = Path("experiments/results/baseline")
        out_res_dir.mkdir(parents=True, exist_ok=True)

        for split_name, df, x_split, y_split in [
            ("val", self.val_df, x_val, y_val),
            ("test", self.test_df, x_test, y_test),
        ]:
            probs = best_clf.predict_proba(x_split)[:, 1]
            preds = (probs >= 0.5).astype(int)

            metrics = compute_classification_metrics(y_split.tolist(), preds.tolist(), probs.tolist())
            results[f"{split_name}_metrics"] = metrics

            if save_artifacts:
                metrics_file = out_res_dir / f"{split_name}_metrics.json"
                metrics_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

                # Token count ước lượng
                n_tokens_list = [len(tokenize(t)) for t in df["text_clean"]]
                pred_df = pd.DataFrame(
                    {
                        "uid": df["uid"].values,
                        "text": df["text_clean"].values,
                        "label": y_split,
                        "rating": df["rating"].values,
                        "prob": np.round(probs, 4),
                        "pred": preds,
                        "n_tokens": n_tokens_list,
                        "n_unk": [0] * len(df),
                        "was_truncated": [False] * len(df),
                    }
                )
                pred_file = out_res_dir / f"predictions_{split_name}.csv"
                pred_df.to_csv(pred_file, index=False)
                logger.info(f"Đã lưu dự đoán {split_name} tại {pred_file}")

        return results


