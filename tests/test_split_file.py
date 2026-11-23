"""Kiểm file split đã commit (chép từ Lab 3): đúng số dòng, không trùng uid, không nhóm nào ở cả hai phía.

Không cần dữ liệu thô, chỉ đọc data/splits/train_val_split.json.
"""
from src.data.split import read_split


def test_committed_split_matches_lab3():
    s = read_split()
    assert (s["n_train"], s["n_val"]) == (22_499, 2_501)
    assert len(s["train_uids"]) == s["n_train"] and len(s["val_uids"]) == s["n_val"]
    assert s["groups_in_both"] == 0
    assert s["seed"] == 42


def test_committed_split_has_disjoint_uids_from_official_train():
    s = read_split()
    train, val = set(s["train_uids"]), set(s["val_uids"])
    assert not train & val
    assert len(train | val) == 25_000
    assert all(u.startswith("train_") for u in train | val)
