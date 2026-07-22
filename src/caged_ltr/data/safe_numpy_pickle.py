"""Narrow, non-executing decoder for legacy NumPy ndarray pickle artifacts."""

from __future__ import annotations

import pickletools
from pathlib import Path

import numpy as np

from caged_ltr.reproducibility import sha256_file

_ALLOWED_OPCODES = {
    "BINBYTES",
    "BININT",
    "BININT1",
    "BININT2",
    "BINGET",
    "BUILD",
    "FRAME",
    "MARK",
    "MEMOIZE",
    "NEWFALSE",
    "NEWTRUE",
    "NONE",
    "PROTO",
    "REDUCE",
    "SHORT_BINBYTES",
    "SHORT_BINUNICODE",
    "STACK_GLOBAL",
    "STOP",
    "TUPLE",
    "TUPLE1",
    "TUPLE2",
    "TUPLE3",
}
_ALLOWED_STRINGS = {
    "<",
    "=",
    ">",
    "_reconstruct",
    "dtype",
    "f4",
    "f8",
    "ndarray",
    "numpy",
    "numpy._core.multiarray",
    "numpy.core.multiarray",
}


def decode_numpy_ndarray_pickle(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> np.ndarray:
    """Decode the exact plain-ndarray pickle form without invoking pickle globals.

    The decoder treats pickle as a bytecode container, rejects every opcode and global
    name outside the minimal NumPy ndarray representation, and reconstructs the numeric
    buffer directly with ``numpy.frombuffer``. It intentionally does not call
    ``pickle.load`` or execute ``REDUCE``/``BUILD`` instructions.
    """
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("legacy ndarray pickle SHA-256 does not match the audited source")
    payload = path.read_bytes()
    operations = list(pickletools.genops(payload))
    if not operations or operations[-1][0].name != "STOP":
        raise ValueError("pickle stream is incomplete")
    disallowed = {opcode.name for opcode, _, _ in operations} - _ALLOWED_OPCODES
    if disallowed:
        raise ValueError(f"pickle contains unsupported opcodes: {sorted(disallowed)}")

    strings = [
        argument
        for opcode, argument, _ in operations
        if opcode.name == "SHORT_BINUNICODE"
    ]
    if set(strings) - _ALLOWED_STRINGS:
        raise ValueError("pickle references names outside the NumPy ndarray allowlist")
    required = {"_reconstruct", "numpy", "ndarray", "dtype"}
    if not required.issubset(strings):
        raise ValueError("pickle is not the supported plain NumPy ndarray representation")
    if sum(opcode.name == "STACK_GLOBAL" for opcode, _, _ in operations) != 3:
        raise ValueError("pickle has an unexpected number of global references")

    shape: tuple[int, int] | None = None
    for index, (opcode, _, _) in enumerate(operations):
        if opcode.name != "TUPLE2" or index < 2:
            continue
        left_op, left, _ = operations[index - 2]
        right_op, right, _ = operations[index - 1]
        integer_ops = {"BININT", "BININT1", "BININT2"}
        if left_op.name in integer_ops and right_op.name in integer_ops:
            candidate = (int(left), int(right))
            if candidate[0] > 0 and candidate[1] > 0:
                shape = candidate
                break
    if shape is None:
        raise ValueError("pickle does not contain a supported two-dimensional shape")

    dtype_tokens = [token for token in strings if token in {"f4", "f8"}]
    endian_tokens = [token for token in strings if token in {"<", "=", ">"}]
    if len(dtype_tokens) != 1 or len(endian_tokens) != 1:
        raise ValueError("pickle must declare one supported floating-point dtype")
    dtype = np.dtype(endian_tokens[0] + dtype_tokens[0])
    expected_bytes = int(np.prod(shape)) * dtype.itemsize
    buffers = [
        bytes(argument)
        for opcode, argument, _ in operations
        if opcode.name in {"BINBYTES", "SHORT_BINBYTES"}
        and isinstance(argument, bytes)
        and len(argument) == expected_bytes
    ]
    if len(buffers) != 1:
        raise ValueError("pickle does not contain exactly one shape-compatible data buffer")

    array = np.frombuffer(buffers[0], dtype=dtype).reshape(shape).copy()
    if not np.isfinite(array).all():
        raise ValueError("semantic array contains non-finite values")
    return array
