"""Test cho Task D3: Nạp và lọc GloVe theo từ vựng."""
import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.data.vocab import Vocab
from src.models.glove import build_glove_matrix


def test_build_glove_matrix_with_mock_glove():
    vocab = Vocab({"<pad>": 0, "<unk>": 1, "great": 2, "movie": 3, "oovword": 4})
    embed_dim = 4

    with tempfile.TemporaryDirectory() as tmp_dir:
        glove_txt = Path(tmp_dir) / "mock_glove.txt"
        # Ghi 2 từ có trong vocab và 1 từ không có
        content = "great 0.1 0.2 0.3 0.4\nmovie 0.5 0.6 0.7 0.8\nother 0.9 1.0 1.1 1.2\n"
        glove_txt.write_text(content, encoding="utf-8")

        npy_path = Path(tmp_dir) / "glove_vocab.npy"
        matrix, coverage = build_glove_matrix(
            vocab=vocab,
            glove_path=glove_txt,
            embed_dim=embed_dim,
            output_path=npy_path,
        )

        assert matrix.shape == (5, embed_dim)
        # Pad token phải toàn số 0
        assert np.allclose(matrix[0], np.zeros(embed_dim))
        # Từ 'great' và 'movie' phải khớp đúng vector trong file
        assert np.allclose(matrix[2], [0.1, 0.2, 0.3, 0.4])
        assert np.allclose(matrix[3], [0.5, 0.6, 0.7, 0.8])
        # Từ oovword và <unk> không có trong file -> coverage là 2 / 4 = 0.5 (không tính <pad>)
        assert coverage == 0.5
        # File .npy phải tồn tại và load được
        assert npy_path.exists()
        loaded = np.load(npy_path)
        assert np.allclose(loaded, matrix)
