from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from caged_ltr.data.safe_numpy_pickle import decode_numpy_ndarray_pickle
from caged_ltr.reproducibility import sha256_file


def test_safe_decoder_reconstructs_plain_numpy_array(tmp_path: Path) -> None:
    source = tmp_path / "array.pkl"
    expected = np.arange(12, dtype=np.float64).reshape(3, 4)
    source.write_bytes(pickle.dumps(expected, protocol=4))

    decoded = decode_numpy_ndarray_pickle(source, expected_sha256=sha256_file(source))

    np.testing.assert_array_equal(decoded, expected)
    assert decoded.flags.owndata


def test_safe_decoder_rejects_non_numpy_global(tmp_path: Path) -> None:
    source = tmp_path / "unsafe.pkl"
    source.write_bytes(pickle.dumps(Path("do-not-execute"), protocol=4))

    with pytest.raises(ValueError, match="unsupported opcodes|outside"):
        decode_numpy_ndarray_pickle(source)
