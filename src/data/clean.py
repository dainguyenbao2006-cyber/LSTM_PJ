"""Làm sạch văn bản và băm nội dung để tìm review trùng.

Lấy từ Lab-DL `labs/lab3/preprocessing/preprocessing.py` (nhánh feat/lab3-redo), giữ nguyên logic,
để số review trùng đếm ra khớp với Lab 3 (123 review test trùng với train, README Lab 3).

Chưa lowercase ở đây: lowercase + tokenize thuộc bước tokenizer (GloVe 6B là uncased).
"""
from __future__ import annotations

import hashlib
import re

_BR_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Thay thẻ <br /> bằng khoảng trắng và gộp khoảng trắng thừa; giữ nguyên phần còn lại.

    Không bỏ stopword vì sẽ mất not/no/never — tín hiệu phủ định mà mô hình chuỗi cần.
    """
    return _SPACE_RE.sub(" ", _BR_RE.sub(" ", text)).strip()


def text_hash(text: str) -> str:
    """SHA-1 của văn bản đã làm sạch; hai review cùng hash = trùng nội dung."""
    return hashlib.sha1(clean_text(text).encode("utf-8")).hexdigest()
