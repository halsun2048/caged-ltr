from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from caged_ltr.data.safe_scipy_pickle import decode_scipy_coo_pickle
from caged_ltr.reproducibility import sha256_file


def test_safe_decoder_reconstructs_scipy_coo(tmp_path: Path) -> None:
    source = tmp_path / "matrix.pkl"
    expected = coo_matrix(
        (
            np.array([1.0, 2.0], dtype=np.float64),
            (
                np.array([0, 2], dtype=np.int32),
                np.array([1, 3], dtype=np.int32),
            ),
        ),
        shape=(3, 4),
    )
    source.write_bytes(pickle.dumps(expected, protocol=4))

    decoded = decode_scipy_coo_pickle(source, expected_sha256=sha256_file(source))

    np.testing.assert_array_equal(decoded.toarray(), expected.toarray())


def test_safe_decoder_rejects_unexpected_global(tmp_path: Path) -> None:
    source = tmp_path / "unsafe.pkl"
    source.write_bytes(pickle.dumps(Path("do-not-execute"), protocol=4))

    with pytest.raises(ValueError, match="unsupported opcodes|outside|supported"):
        decode_scipy_coo_pickle(source)
