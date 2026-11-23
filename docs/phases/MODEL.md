# MODEL.md — Định nghĩa mô hình

- **Motivation/Background**: Các mô hình phải chỉ khác nhau đúng một biến để so sánh công bằng.
- **Purpose**: Định nghĩa Embedding + LSTM/GRU/BiLSTM + Linear, nạp GloVe, và (mở rộng) attention pooling.
- **Overview Pipeline**: Phạm vi → đặc tả → artifact → kiểm chứng.
- **Detailed Plan**: Header; Goals; Input/Output; How to do it; Detailed plan / gotchas; Verification gates; Links.
- **References**: [OVERVIEW.md](../OVERVIEW.md), [PURPOSE.md](../PURPOSE.md) §7, [MODELS.md](../MODELS.md).
- **Created**: 2026-11-05T11:21:04+07:00
- **Last Updated**: 2026-11-23T10:02:29+07:00

---

## Header

- **Title:** Định nghĩa mô hình
- **Date created:** 2026-11-05
- **Last updated:** 2026-11-23
- **Description:** Định nghĩa Embedding + LSTM/GRU/BiLSTM + Linear và nạp GloVe.
- **Status:** To Do
- **Owner:** Đại
- **Hạn:** T2 23/11 tối (code + test); pilot sáng T3 24/11

## Goals / Purpose

- Xong: một class `SentimentRNN` tham số hóa `rnn_type`, `bidirectional`, `pretrained_embeddings`, `freeze`; test pass; số tham số điền vào [MODELS.md](../MODELS.md).
- Mở rộng (nếu kịp): tuỳ chọn `pooling="attention"` cho E1.
- Không làm: DistilBERT trong class này (E3 viết riêng).

## Input / Output

- **Input:** `data/processed/reviews.parquet` của Bảo (cột `uid, split, text_clean, label, ...`, đã loại trùng, split Lab 3), `data/glove/glove.6B.100d.txt` (trên Kaggle dùng dataset GloVe có sẵn).
- **Đại viết:** tokenizer (lowercase vì GloVe 6B uncased; giữ `not`, `n't`, `!`, `?`; không bỏ stopword), vocab chỉ từ train, Dataset/DataLoader (`src/data/tokenizer.py`, `src/data/vocab.py`, `src/data/dataset.py`).
- **Output:** `src/models/rnn.py`, `src/models/glove.py`, `data/processed/glove_100d_vocab.npy`, `tests/test_models.py`.

## How to do it (general plan)

1. Viết class `SentimentRNN` theo [MODELS.md](../MODELS.md) §2 (Embedding 100 → RNN 128 → Dropout → Linear).
2. Viết hàm dựng ma trận embedding từ GloVe theo vocab, lưu `.npy`; ghi tỷ lệ phủ vào [DATA.md](../DATA.md) §6.
3. Test: input `[B, L]` + lengths → output `[B, 1]`; overfit một batch nhỏ; save/load checkpoint cho cùng output.

## Detailed plan / gotchas

- Dùng `pack_padded_sequence` với độ dài thật.
- BiLSTM: nối `h_n[-2]` và `h_n[-1]`, không lấy output tại vị trí padding.
- `padding_idx=0`; token lạ `<unk>=1`.
- Embedding dim phải bằng chiều GloVe (100).

## Verification gates

- [ ] `pytest tests/test_models.py` pass (shape, overfit 1 batch, save/load).
- [ ] Overfit một batch 32 mẫu tới loss gần 0.
- [ ] M4a: số tham số train được không gồm embedding.

## Links

- Progress tracking: [../progress/MODEL_STATUS.md](../progress/MODEL_STATUS.md)
- Related: [OVERVIEW.md](../OVERVIEW.md)
