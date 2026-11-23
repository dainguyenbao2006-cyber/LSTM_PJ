"""Test phần dữ liệu lấy từ Lab 3 (chạy offline trên cây aclImdb giả).

Phần lớn lấy từ Lab-DL `labs/lab3/tests/test_preprocessing.py` và `test_eda.py` (nhánh feat/lab3-redo).
"""
import numpy as np
import pandas as pd
import pytest

from src.data.clean import clean_text, text_hash
from src.data.download import load_split, read_movie_ids
from src.data.eda import duplicate_counts, length_summary, negation_share, run_eda, token_lengths
from src.data.split import apply_split, make_train_val_split, split_groups


# ---------------------------------------------------------------- đọc dữ liệu
def test_load_split_reads_id_rating_label_movie(fake_imdb_root):
    df = load_split("train", fake_imdb_root)
    assert len(df) == 8
    assert set(df.columns) >= {"uid", "split", "id", "rating", "label", "movie_id", "text"}
    row = df.set_index("uid").loc["train_neg_3"]
    assert (row["rating"], row["label"], row["movie_id"]) == (4, 0, "tt0000006")
    assert set(df.loc[df["label"] == 1, "rating"]) <= {7, 8, 9, 10}
    assert set(df.loc[df["label"] == 0, "rating"]) <= {1, 2, 3, 4}


def test_load_split_writes_and_reuses_parquet_cache(fake_imdb_root):
    first = load_split("test", fake_imdb_root)
    assert (fake_imdb_root / "imdb_test.parquet").exists()
    pd.testing.assert_frame_equal(first, load_split("test", fake_imdb_root))


def test_load_split_stops_when_url_lines_do_not_match(fake_imdb_root):
    urls = fake_imdb_root / "aclImdb" / "train" / "urls_pos.txt"
    urls.write_text("\n".join(urls.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="URL lines"):
        load_split("train", fake_imdb_root)


def test_read_movie_ids_rejects_line_without_movie(tmp_path):
    p = tmp_path / "urls.txt"
    p.write_text("http://www.imdb.com/title/tt0453418/usercomments\nhttp://example.com/x\n")
    with pytest.raises(ValueError, match="no movie id"):
        read_movie_ids(p)


# ---------------------------------------------------------------- làm sạch
def test_clean_text_removes_br_only():
    assert clean_text("Great.<br /><br />I did NOT like   it<BR/>!") == "Great. I did NOT like it !"
    assert clean_text("no tags, don't change") == "no tags, don't change"


def test_text_hash_ignores_br_and_spaces():
    assert text_hash("a<br />b") == text_hash("a b") == text_hash("a   b")
    assert text_hash("a b") != text_hash("A b")


# ---------------------------------------------------------------- chia train / val
def test_split_groups_merge_same_movie_and_duplicate_text(fake_imdb_root):
    df = load_split("train", fake_imdb_root).set_index("uid")
    g = split_groups(df.reset_index()).set_axis(df.index)
    assert g["train_pos_0"] == g["train_pos_1"]          # cùng phim tt0000001
    assert g["train_pos_2"] == g["train_neg_2"]          # khác phim nhưng trùng nội dung
    assert g["train_pos_0"] != g["train_neg_3"]


def test_make_train_val_split_has_no_group_in_both(fake_imdb_root):
    train = load_split("train", fake_imdb_root)
    s = make_train_val_split(train, seed=0, n_folds=2)
    assert s["groups_in_both"] == 0
    assert s["n_train"] + s["n_val"] == len(train)
    assert not set(s["train_uids"]) & set(s["val_uids"])


def test_apply_split_returns_rows_in_split_order(fake_imdb_root):
    train_all = load_split("train", fake_imdb_root)
    s = make_train_val_split(train_all, seed=0, n_folds=2)
    train, val = apply_split(train_all, s)
    assert train["uid"].tolist() == s["train_uids"] and val["uid"].tolist() == s["val_uids"]


def test_apply_split_stops_on_unknown_uid(fake_imdb_root):
    train_all = load_split("train", fake_imdb_root)
    with pytest.raises(ValueError, match="not found"):
        apply_split(train_all, {"train_uids": ["train_pos_999"], "val_uids": []})


# ---------------------------------------------------------------- EDA
def test_length_summary_percentiles_and_over_limits():
    lengths = np.array([10, 100, 201, 450, 700])
    s = length_summary(lengths, limits=(200, 400, 600))
    assert s["median"] == 201 and s["max"] == 700
    assert (s["pct_over_200"], s["pct_over_400"], s["pct_over_600"]) == (60.0, 40.0, 20.0)


def test_token_lengths_default_counts_words():
    assert token_lengths(["great movie", "the film was bad"]).tolist() == [2, 4]


def test_negation_share():
    assert negation_share(["not good", "I don't know", "great", "no way"]) == 75.0
    assert negation_share(["knot", "nobody came"]) == 50.0  # "knot" không phải phủ định


def test_duplicate_counts(fake_imdb_root):
    train, test = load_split("train", fake_imdb_root), load_split("test", fake_imdb_root)
    d = duplicate_counts(train, test)
    assert d["train_extra_copies"] == 1           # "Very good film." 2 bản
    assert d["test_reviews_also_in_train"] == 1   # "Good acting and good plot."
    assert d["train_dup_groups_label_conflict"] == 1


def test_run_eda_writes_files_for_each_split(fake_imdb_root, tmp_path):
    train_all, test = load_split("train", fake_imdb_root), load_split("test", fake_imdb_root)
    train, val = apply_split(train_all, make_train_val_split(train_all, seed=0, n_folds=2))
    out = run_eda({"train": train, "val": val, "test": test}, tmp_path / "results", tmp_path / "plots")
    for p in out["paths"].values():
        assert p.exists() and p.stat().st_size > 0
    assert out["summary"]["split"].tolist() == ["train", "val", "test"]
    assert out["summary"]["n_reviews"].sum() == len(train_all) + len(test)
