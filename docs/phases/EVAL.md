# EVAL.md — Đánh giá, phân tích lỗi & giải thích mô hình

- **Motivation/Background**: Môn học yêu cầu so sánh với baseline và phân tích lỗi, giới hạn; đăng ký cam kết số liệu riêng cho từng nhóm tình huống (M4).
- **Purpose**: Đánh giá test một lần cho mỗi checkpoint cuối, so sánh với baseline, phân tích lỗi 7 nhóm tình huống, giải thích mô hình.
- **Overview Pipeline**: Phạm vi → đặc tả → artifact → kiểm chứng.
- **Detailed Plan**: Header; Goals; Input/Output; How to do it; Detailed plan / gotchas; Verification gates; Links.
- **References**: [OVERVIEW.md](../OVERVIEW.md), [PURPOSE.md](../PURPOSE.md) §4, §7, §9.
- **Created**: 2026-11-05T11:21:04+07:00
- **Last Updated**: 2026-11-23T10:13:39+07:00

---

## Header

- **Title:** Đánh giá, phân tích lỗi & giải thích mô hình
- **Date created:** 2026-11-05
- **Last updated:** 2026-11-23
- **Description:** Đánh giá test một lần, so sánh với baseline, phân tích lỗi 7 nhóm, giải thích mô hình.
- **Status:** To Do
- **Owner:** Đại (chạy test một lần, metric, McNemar, xuất file dự đoán); Thư (phân tích lỗi, giải thích mô hình, giới hạn)
- **Hạn:** T3 24/11 tối

## Goals / Purpose

- Xong: bảng so sánh B0/B1/M1–M4b (mean ± std 3 seed, kèm số tham số, thời gian train/inference, McNemar mô hình tốt nhất vs B1, dòng DistilBERT trích Lab 3) trong [experiments/README.md](../experiments/README.md); confusion matrix; số liệu từng nhóm tình huống, mỗi nhóm ≥ 10 ví dụ sai kèm giải thích; kết quả giải thích mô hình.
- Không làm: sửa mô hình dựa trên kết quả test.

## Input / Output

- **Input:** checkpoint cuối mỗi mô hình (3 seed), `data/processed/reviews.parquet` (split test), `data/sample/handwritten_30.csv` (Thư viết).
- **Bàn giao Đại → Thư:** `experiments/results/<exp>/predictions_<split>.csv`, mỗi dòng một review, cột `uid, text, label, rating, prob, pred, n_tokens, n_unk, was_truncated`. Thư phân tích lỗi từ file này, không chạy lại mô hình.
- **Output:**
  - `experiments/results/<exp>/seed_<s>/test_metrics.json`, `experiments/results/<exp>/test_summary.json`.
  - File dự đoán: chỉ dùng một định dạng `predictions_<split>.csv` ở mục bàn giao trên (cột `case_type` do Thư thêm khi phân tích lỗi, lưu ở `experiments/results/error_analysis/`).
  - `experiments/results/comparison_val.csv`, `experiments/results/comparison_test.csv` (bảng so sánh mọi mô hình), `experiments/results/mcnemar.json`.
  - `experiments/results/error_analysis/` (bảng theo nhóm, ví dụ sai).
  - `experiments/plots/` (confusion matrix, ROC, heatmap từ quan trọng).

## How to do it (general plan)

1. `src/evaluate.py` gọi `src/eval/metrics.py`: accuracy, precision/recall/F1 từng lớp, macro-F1, ROC-AUC, confusion matrix (FP và FN), thời gian inference.
2. Lưu dự đoán từng review kèm xác suất (`predictions_<split>.csv`).
2b. Bảng so sánh mọi mô hình (val trước, test sau khi chốt); McNemar chính xác giữa mô hình tốt nhất và B1 trên cùng 25.000 review test; thêm dòng DistilBERT từ Lab 3 (ghi rõ nguồn).
3. Gán nhóm tình huống (`src/eval/error_analysis.py`) theo [PURPOSE.md](../PURPOSE.md) §9:
   1. Độ dài: < 100, 100–300, 300–600, > 600 từ (xem cắt ngắn có mất câu kết luận).
   2. Phủ định.
   3. Khen chê lẫn lộn, đổi giọng ở cuối.
   4. Mỉa mai, châm biếm.
   5. Điểm sát ngưỡng (4, 7) so với cực trị (1, 10).
   6. Nhiều từ OOV / từ hiếm.
   7. Bộ ~30 câu tự viết (`handwritten_30.csv`).
4. Tính metric riêng từng nhóm; chọn ≥ 10 ví dụ sai mỗi nhóm, viết giải thích.
5. Giải thích mô hình (`src/eval/explain.py`): occlusion hoặc gradient × embedding; kiểm tra "đường tắt" (tên diễn viên, độ dài).
6. (Mở rộng E4) đánh giá chéo Rotten Tomatoes với checkpoint có sẵn.
7. Ghi giới hạn của mô hình và dữ liệu.

## Detailed plan / gotchas

- Chọn checkpoint theo val, cố định trước khi xem test.
- Phân tích lỗi trên test chỉ để mô tả, không quay lại tinh chỉnh.
- Nhóm 2–4 cần quy tắc gán (từ khoá/heuristic) hoặc gán tay; ghi rõ cách gán.

## Verification gates

- [ ] Mỗi số trong bảng có file `test_metrics.json` nguồn.
- [ ] Confusion matrix cộng đủ 25.000.
- [ ] Đủ 7 nhóm, mỗi nhóm ≥ 10 ví dụ sai có giải thích.

## Links

- Progress tracking: [../progress/EVAL_STATUS.md](../progress/EVAL_STATUS.md)
- Related: [OVERVIEW.md](../OVERVIEW.md)
