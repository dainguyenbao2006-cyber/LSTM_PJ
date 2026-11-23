"""Các hàm tính độ đo đánh giá và kiểm định thống kê (Metrics & Statistical Tests).

Theo đặc tả docs/phases/EVAL.md:
- Accuracy, Precision / Recall / F1 từng lớp, Macro-F1, ROC-AUC.
- Confusion Matrix (TN, FP, FN, TP).
- Đo thời gian suy luận (latency per sample, latency per batch).
- Kiểm định McNemar (chính xác hoặc hiệu chỉnh Edwards) so sánh 2 mô hình.
"""
from __future__ import annotations

import time
from typing import Any
import numpy as np
import scipy.stats as stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch


def compute_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Tính toán toàn bộ các độ đo phân loại cần thiết."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_t, y_p))
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))

    # Precision, Recall, F1 từng lớp (0: tiêu cực, 1: tích cực)
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_t, y_p, labels=[0, 1], zero_division=0
    )

    # Confusion matrix: tn, fp, fn, tp
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    res: dict[str, Any] = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "negative_class": {
            "precision": round(float(precisions[0]), 4),
            "recall": round(float(recalls[0]), 4),
            "f1": round(float(f1s[0]), 4),
            "support": int(supports[0]),
        },
        "positive_class": {
            "precision": round(float(precisions[1]), 4),
            "recall": round(float(recalls[1]), 4),
            "f1": round(float(f1s[1]), 4),
            "support": int(supports[1]),
        },
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "total": tn + fp + fn + tp,
        },
    }

    if y_prob is not None:
        y_pr = np.asarray(y_prob, dtype=float)
        try:
            auc = float(roc_auc_score(y_t, y_pr))
            res["roc_auc"] = round(auc, 4)
        except Exception:
            res["roc_auc"] = None

    return res


def measure_inference_speed(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Đo thời gian suy luận (latency) trên mỗi batch và mỗi mẫu."""
    model.eval()
    total_samples = 0
    total_batches = 0

    # Warmup
    with torch.no_grad():
        for batch in dataloader:
            _ = model(batch["input_ids"].to(device), batch["lengths"])
            break

    start_t = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            _ = model(batch["input_ids"].to(device), batch["lengths"])
            total_samples += batch["lengths"].size(0)
            total_batches += 1
    total_elapsed = time.perf_counter() - start_t

    ms_per_sample = (total_elapsed / max(1, total_samples)) * 1000.0
    ms_per_batch = (total_elapsed / max(1, total_batches)) * 1000.0

    return {
        "total_seconds": round(total_elapsed, 4),
        "latency_ms_per_sample": round(ms_per_sample, 3),
        "latency_ms_per_batch": round(ms_per_batch, 2),
        "samples_per_second": round(total_samples / max(0.001, total_elapsed), 1),
    }


def mcnemar_test(
    y_true: list[int] | np.ndarray,
    y_pred_a: list[int] | np.ndarray,
    y_pred_b: list[int] | np.ndarray,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> dict[str, Any]:
    """Thực hiện kiểm định thống kê McNemar giữa hai mô hình trên cùng tập kiểm thử.

    b: Model A đúng, Model B sai
    c: Model A sai, Model B đúng
    Stat = (|b - c| - 1)^2 / (b + c) (hiệu chỉnh liên tục)
    """
    y_t = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(y_pred_a, dtype=int)
    pred_b = np.asarray(y_pred_b, dtype=int)

    correct_a = pred_a == y_t
    correct_b = pred_b == y_t

    n_both_correct = int(np.sum(correct_a & correct_b))
    n_both_wrong = int(np.sum((~correct_a) & (~correct_b)))
    b = int(np.sum(correct_a & (~correct_b)))  # A đúng, B sai
    c = int(np.sum((~correct_a) & correct_b))  # A sai, B đúng
    total = len(y_t)

    table = [
        [n_both_correct, b],
        [c, n_both_wrong],
    ]

    discordant = b + c
    if discordant == 0:
        stat = 0.0
        p_val = 1.0
    else:
        # Edwards continuity correction
        stat = float(((abs(b - c) - 1.0) ** 2) / discordant)
        p_val = float(stats.chi2.sf(stat, df=1))

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "contingency_table": {
            "both_correct": n_both_correct,
            "a_correct_b_wrong": b,
            "a_wrong_b_correct": c,
            "both_wrong": n_both_wrong,
            "total": total,
        },
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "significant_p05": bool(p_val < 0.05),
        "significant_p01": bool(p_val < 0.01),
        "interpretation": (
            f"Sự khác biệt giữa {model_a_name} và {model_b_name} có ý nghĩa thống kê (p < 0.05)"
            if p_val < 0.05
            else f"Không đủ bằng chứng kết luận sự khác biệt giữa {model_a_name} và {model_b_name} (p >= 0.05)"
        ),
    }
