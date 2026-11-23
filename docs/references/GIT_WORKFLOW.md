# GIT_WORKFLOW.md — Quy trình Git cho nhóm 3 người

- **Motivation/Background**: Ba người cùng sửa một repo dễ ghi đè code nhau và commit nhầm dữ liệu lớn.
- **Purpose**: Quy ước nhánh, commit, PR, điều cấm và các bước làm một task.
- **Overview Pipeline**: Pull main → nhánh mới → commit nhỏ → pytest → PR → một bạn khác review → Thư merge.
- **Detailed Plan**: §1 Nhánh; §2 Commit; §3 Pull request; §4 Điều cấm; §5 Các bước làm một task.
- **References**: [../../README.md](../../README.md), [../TASKS.md](../TASKS.md)
- **Created**: 2026-11-05T11:21:04+07:00
- **Last Updated**: 2026-11-23T10:36:35+07:00

---

## 1. Nhánh

- `main` luôn chạy được; **không push thẳng lên `main`**, mọi thay đổi vào `main` qua PR.
- Trước khi tạo nhánh mới: `git checkout main && git pull`.
- Một nhánh cho một việc:

| Loại việc | Tên nhánh | Ví dụ |
| :--- | :--- | :--- |
| Code mới | `feat/<phase>-<viec>` | `feat/data-prepare`, `feat/model-rnn` |
| Sửa lỗi | `fix/<loi>` | `fix/pack-padded-length` |
| Tài liệu | `docs/<tai-lieu>` | `docs/data-md` |
| Thí nghiệm (chạy, log, kết quả) | `exp/<NN>-<ten>` | `exp/02-lstm`, `exp/05-bilstm-glove` |

- Mã thí nghiệm `<NN>` theo [../experiments/README.md](../experiments/README.md) (EXP-01 baseline … EXP-07 `max_len`).

## 2. Commit

Theo Conventional Commits: `<type>(<scope>): <mô tả ngắn>` — **bắt buộc có scope**.

- `type`: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `exp`.
- `scope`: `data`, `baseline`, `model`, `train`, `eval`, `demo`, `docs`.
- Mô tả tiếng Việt, ngắn, nói việc đã làm. Commit có nhiều thay đổi thì thêm phần thân (dòng trống rồi gạch đầu dòng).
- Ví dụ: `feat(data): loại review trùng`, `test(model): overfit 1 batch`, `exp(train): chạy LSTM 3 seed`, `docs(docs): điền DATA.md §4`.
- Không thêm dòng `Co-Authored-By` của công cụ AI; việc dùng AI khai trong báo cáo.

## 3. Pull request

- Chạy `python -m pytest` trước khi mở PR; ghi kết quả vào mô tả.
- Mô tả PR theo mẫu:

```
## Làm gì
## Vì sao
## Cách kiểm tra
(lệnh đã chạy + kết quả, ví dụ: python -m pytest → 20 passed)
## Progress đã cập nhật
(file docs/progress/<PHASE>_STATUS.md, trạng thái task trong docs/TASKS.md)
```

- Review: Bảo và Đại review chéo PR của nhau; PR của Thư do Bảo hoặc Đại review. Thư duyệt và merge mọi PR vào `main`. Mỗi người phải tự giải thích được code mình viết.
- PR nhỏ, một việc; không gộp code của nhiều task vào một PR.

## 4. Điều cấm

- Commit `data/raw/`, `data/splits/`, `data/glove/`, `data/processed/`, checkpoint (`*.pt`, `*.pth`, `*.joblib`), `.venv/`.
  - Được commit: `data/sample/` và **`data/splits/train_val_split.json`** (chỉ chứa danh sách uid, chép từ Lab 3, dùng chung cho mọi thí nghiệm — [../DATA.md](../DATA.md) §3).
- Commit đường dẫn tuyệt đối (`C:\Users\...`), **token Kaggle** (`kaggle.json`, `access_token`), API key.
- Sửa số trong tài liệu mà không có file kết quả đi kèm.
- Push thẳng lên `main`; force push lên nhánh người khác đang dùng.

## 5. Các bước làm một task

1. Nhận task trong [../TASKS.md](../TASKS.md), đổi trạng thái sang 🟨.
2. `git checkout main && git pull` → `git checkout -b feat/<phase>-<viec>`.
3. Code + test; commit nhỏ theo §2.
4. `python -m pytest` pass.
5. `git push -u origin <nhánh>` → mở PR theo mẫu §3.
6. Ghi bằng chứng vào `docs/progress/<PHASE>_STATUS.md`, đổi trạng thái task sang ✅ (trong cùng PR).
7. Sau khi merge: quay lại bước 2 cho task tiếp theo.
