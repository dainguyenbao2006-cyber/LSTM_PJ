# DATA_PREP_STATUS.md — Tiến độ: Chuẩn bị dữ liệu & split

- **Motivation/Background**: Theo dõi nhanh phase Chuẩn bị dữ liệu & split mà không cần đọc lại toàn bộ đặc tả.
- **Purpose**: Ghi việc đã xong kèm bằng chứng, blocker, quyết định và bước tiếp theo.
- **Overview Pipeline**: Log → blocker → quyết định → bước tiếp theo.
- **Detailed Plan**: Header; Checklist; Log; Blockers; Decisions; Next step; Links.
- **References**: [../phases/DATA_PREP.md](../phases/DATA_PREP.md), [../OVERVIEW.md](../OVERVIEW.md).
- **Created**: 2026-11-05T11:21:04+07:00
- **Last Updated**: 2026-11-23T10:02:29+07:00

---

## Header

- **Title:** Chuẩn bị dữ liệu & split
- **Date created:** 2026-11-05
- **Last updated:** 2026-11-23
- **Status:** To Do
- **Owner:** Bảo
- **Hạn:** T2 23/11 tối — giao `reviews.parquet` cho Đại
- **Phase doc:** [../phases/DATA_PREP.md](../phases/DATA_PREP.md)

## Checklist

Chỉ đánh dấu `[x]` khi có bằng chứng (lệnh, đường dẫn output, commit).

- [ ] Đọc xong đặc tả phase
- [ ] Code chạy được trên tập con
- [ ] Verification gates trong phase doc đều pass
- [ ] Cập nhật tài liệu liên quan (DATA.md §3–§4)

## Log

- 2026-11-05: Tạo file.
- 2026-11-23: Dựng repo (nhánh `chore/repo-setup`): chép từ Lab 3 phần tải/đọc IMDb, làm sạch, hash, file split, hàm EDA + test (`python -m pytest`: 17 passed). Chạy thử `python -m src.data.eda` trên dữ liệu thật: 22.499 / 2.501 / 25.000 review, 123 review test trùng train. Kết quả chạy thử không commit — Bảo chạy lại sau khi loại trùng.

## Blockers

- Chưa có.

## Decisions

- Chưa có.

## Next step

- Bắt đầu bước 1 trong [../phases/DATA_PREP.md](../phases/DATA_PREP.md).

## Links

- Phase doc: [../phases/DATA_PREP.md](../phases/DATA_PREP.md)
- Experiments: [../experiments/README.md](../experiments/README.md)
