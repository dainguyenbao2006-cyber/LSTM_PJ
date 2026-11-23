"""Đánh giá mô hình trên tập kiểm thử (Test Evaluation) và xuất kết quả.

Theo đặc tả trong docs/phases/EVAL.md và docs/TASKS.md §3 D8:
- Nạp checkpoint cuối (best.pt) và vocab.json.
- Chạy đánh giá trên tập test (hoặc val).
- Xuất file dự đoán predictions_<split>.csv với đầy đủ các cột bàn giao:
  uid, text, label, rating, prob, pred, n_tokens, n_unk, was_truncated
- Xuất test_metrics.json, đo latency, vẽ confusion matrix và ROC curve.
- Xuất bảng so sánh tổng hợp comparison_*.csv và kiểm định McNemar (mcnemar.json).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
import torch

from src.data.dataset import IMDbDataset, create_dataloader, load_imdb_split_data
from src.data.vocab import Vocab
from src.eval.metrics import (
    compute_classification_metrics,
    mcnemar_test,
    measure_inference_speed,
)
from src.models.rnn import SentimentRNN
from src.utils.paths import experiment_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def evaluate_split(
    checkpoint_path: str | Path,
    vocab_path: str | Path,
    split: str = "test",
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    device_name: str | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Đánh giá checkpoint trên split chỉ định và xuất đầy đủ artifacts."""
    ckpt_file = Path(checkpoint_path)
    vocab_file = Path(vocab_path)

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại {ckpt_file}")
    if not vocab_file.exists():
        raise FileNotFoundError(f"Không tìm thấy vocab tại {vocab_file}")

    # 1. Thiết bị
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Tải Vocab và Mô hình
    vocab = Vocab.load(vocab_file)
    model, ckpt = SentimentRNN.load_checkpoint(str(ckpt_file), device=device)
    model.eval()

    # 3. Đọc dữ liệu split
    df = load_imdb_split_data(parquet_path=data_path, split=split)
    logger.info(f"Đánh giá {len(df)} mẫu từ split '{split}'...")

    max_len = ckpt.get("config", {}).get("data", {}).get("max_len", 250)
    dataset = IMDbDataset(
        texts=df["text_clean"],
        labels=df["label"],
        uids=df["uid"],
        ratings=df["rating"],
        vocab=vocab,
        max_len=max_len,
    )
    dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)

    # 4. Dự đoán toàn bộ split
    all_uids: list[str] = []
    all_texts: list[str] = []
    all_ratings: list[int] = []
    all_targets: list[int] = []
    all_probs: list[float] = []
    all_preds: list[int] = []
    all_n_tokens: list[int] = []
    all_n_unk: list[int] = []
    all_was_truncated: list[bool] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"]

            probs = model.predict_proba(input_ids, lengths).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_uids.extend(batch["uids"])
            all_texts.extend(batch["texts"])
            all_ratings.extend(batch["ratings"])
            all_targets.extend(batch["labels"].numpy().astype(int))
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_n_tokens.extend(batch["n_tokens"])
            all_n_unk.extend(batch["n_unk"])
            all_was_truncated.extend(batch["was_truncated"])

    # 5. Tính toán metrics
    metrics = compute_classification_metrics(all_targets, all_preds, all_probs)
    speed_metrics = measure_inference_speed(model, dataloader, device)
    param_counts = model.count_parameters()

    results_data = {
        "split": split,
        "checkpoint": str(ckpt_file),
        "parameters": param_counts,
        "metrics": metrics,
        "speed": speed_metrics,
    }

    # 6. Xác định thư mục output
    if output_dir:
        out_path = Path(output_dir)
    else:
        # Mặc định theo cấu trúc thư mục của checkpoint
        out_path = ckpt_file.parent

    out_path.mkdir(parents=True, exist_ok=True)

    # 7. Xuất file bàn giao predictions_<split>.csv
    pred_df = pd.DataFrame(
        {
            "uid": all_uids,
            "text": all_texts,
            "label": all_targets,
            "rating": all_ratings,
            "prob": [round(p, 4) for p in all_probs],
            "pred": all_preds,
            "n_tokens": all_n_tokens,
            "n_unk": all_n_unk,
            "was_truncated": all_was_truncated,
        }
    )
    pred_csv_path = out_path / f"predictions_{split}.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    logger.info(f"Đã lưu bảng dự đoán chi tiết ({len(pred_df)} dòng) tại {pred_csv_path}")

    # 8. Lưu metrics JSON
    metrics_json_path = out_path / f"{split}_metrics.json"
    metrics_json_path.write_text(json.dumps(results_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Đã lưu metrics tại {metrics_json_path}")

    # 9. Vẽ Confusion Matrix và ROC Curve
    plots_dir = experiment_dir("plots", "eval")
    plots_dir.mkdir(parents=True, exist_ok=True)
    exp_stem = ckpt_file.parts[-3] if len(ckpt_file.parts) >= 3 else "model"

    # CM plot
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_arr = np.array([
        [metrics["confusion_matrix"]["tn"], metrics["confusion_matrix"]["fp"]],
        [metrics["confusion_matrix"]["fn"], metrics["confusion_matrix"]["tp"]],
    ])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_arr, display_labels=["Negative", "Positive"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix: {exp_stem} ({split})", fontweight="bold")
    cm_plot_path = plots_dir / f"{exp_stem}_{split}_cm.png"
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=150)
    plt.close(fig)

    # ROC plot
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    auc_val = metrics.get("roc_auc", 0.0)
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax.set_title(f"ROC Curve: {exp_stem} ({split})", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    roc_plot_path = plots_dir / f"{exp_stem}_{split}_roc.png"
    plt.tight_layout()
    plt.savefig(roc_plot_path, dpi=150)
    plt.close(fig)

    logger.info(f"Đã xuất đồ thị đánh giá tại {plots_dir}")
    return results_data


def compare_all_models(
    results_base: Path | str = "experiments/results",
    split: str = "test",
) -> pd.DataFrame:
    """Tạo bảng so sánh tổng hợp mọi mô hình (comparison_<split>.csv)."""
    base = Path(results_base)
    rows: list[dict[str, Any]] = []

    for exp_dir in base.iterdir():
        if not exp_dir.is_dir():
            continue
        exp_name = exp_dir.name

        # Tìm các seed metrics
        seed_accs: list[float] = []
        seed_f1s: list[float] = []
        seed_aucs: list[float] = []
        total_p = None
        trainable_p = None
        latency = None

        for s_dir in exp_dir.glob("seed_*"):
            m_file = s_dir / f"{split}_metrics.json"
            if m_file.exists():
                data = json.loads(m_file.read_text(encoding="utf-8"))
                m = data["metrics"]
                seed_accs.append(m["accuracy"])
                seed_f1s.append(m["macro_f1"])
                if m.get("roc_auc") is not None:
                    seed_aucs.append(m["roc_auc"])
                if total_p is None:
                    total_p = data.get("parameters", {}).get("total")
                    trainable_p = data.get("parameters", {}).get("trainable")
                    latency = data.get("speed", {}).get("latency_ms_per_sample")

        if seed_accs:
            acc_mean = float(np.mean(seed_accs))
            acc_std = float(np.std(seed_accs))
            f1_mean = float(np.mean(seed_f1s))
            f1_std = float(np.std(seed_f1s))
            auc_mean = float(np.mean(seed_aucs)) if seed_aucs else None

            rows.append(
                {
                    "model": exp_name,
                    "accuracy": f"{acc_mean * 100:.2f} ± {acc_std * 100:.2f}%",
                    "macro_f1": f"{f1_mean:.4f} ± {f1_std:.4f}",
                    "roc_auc": f"{auc_mean:.4f}" if auc_mean else "—",
                    "total_params": f"{total_p:,}" if total_p else "—",
                    "trainable_params": f"{trainable_p:,}" if trainable_p else "—",
                    "latency_ms_sample": latency if latency else "—",
                }
            )

    df_comp = pd.DataFrame(rows)
    out_csv = base / f"comparison_{split}.csv"
    df_comp.to_csv(out_csv, index=False)
    logger.info(f"Đã lưu bảng so sánh tổng hợp vào {out_csv}")
    return df_comp


def run_mcnemar_cli(
    pred_file_a: str | Path,
    pred_file_b: str | Path,
    model_a_name: str,
    model_b_name: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Chạy kiểm định McNemar giữa 2 file predictions và xuất ra JSON."""
    df_a = pd.read_csv(pred_file_a)
    df_b = pd.read_csv(pred_file_b)

    assert len(df_a) == len(df_b), "Hai file dự đoán phải có cùng số dòng!"
    assert (df_a["uid"] == df_b["uid"]).all(), "Thứ tự các uid trong hai file phải trùng khớp!"

    res = mcnemar_test(
        y_true=df_a["label"].values,
        y_pred_a=df_a["pred"].values,
        y_pred_b=df_b["pred"].values,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
    )

    out_file = Path(output_path) if output_path else Path("experiments/results/mcnemar.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Đã lưu kết quả kiểm định McNemar vào {out_file}")
    return res


