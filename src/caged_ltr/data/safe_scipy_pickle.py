"""Narrow, non-executing decoder for the official RLMRec COO pickle artifacts."""

from __future__ import annotations

import pickletools
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

from caged_ltr.reproducibility import sha256_file

_ALLOWED_OPCODES = {
    "BINGET",
    "BINBYTES",
    "BININT",
    "BININT1",
    "BININT2",
    "BUILD",
    "EMPTY_DICT",
    "EMPTY_TUPLE",
    "FRAME",
    "MARK",
    "MEMOIZE",
    "NEWFALSE",
    "NEWOBJ",
    "NEWTRUE",
    "NONE",
    "PROTO",
    "REDUCE",
    "SETITEMS",
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
    "_reconstruct",
    "_shape",
    "col",
    "coo_matrix",
    "coords",
    "data",
    "dtype",
    "f8",
    "has_canonical_format",
    "i4",
    "maxprint",
    "ndarray",
    "numpy",
    "numpy._core.multiarray",
    "numpy.core.multiarray",
    "row",
    "scipy.sparse._coo",
}
_INTEGER_OPCODES = {"BININT", "BININT1", "BININT2"}


def _next_buffer(operations: list[tuple[object, object, int]], start: int) -> bytes:
    for opcode, argument, _ in operations[start:]:
        if opcode.name in {"BINBYTES", "SHORT_BINBYTES"} and isinstance(argument, bytes):
            return bytes(argument)
    raise ValueError("COO pickle field does not contain a numeric buffer")


def _single_aligned_buffer(
    operations: list[tuple[object, object, int]],
    start: int,
    stop: int,
    *,
    alignment: int,
) -> bytes:
    buffers = [
        bytes(argument)
        for opcode, argument, _ in operations[start:stop]
        if opcode.name in {"BINBYTES", "SHORT_BINBYTES"}
        and isinstance(argument, bytes)
        and len(argument) > 0
        and len(argument) % alignment == 0
    ]
    if len(buffers) != 1:
        raise ValueError("COO field does not contain exactly one aligned numeric buffer")
    return buffers[0]


def decode_scipy_coo_pickle(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> coo_matrix:
    """Decode the audited SciPy COO representation without executing pickle bytecode."""
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("legacy COO pickle SHA-256 does not match the audited source")
    operations = list(pickletools.genops(path.read_bytes()))
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
        raise ValueError("pickle references names outside the audited COO allowlist")
    required = {
        "scipy.sparse._coo",
        "coo_matrix",
        "_shape",
        "data",
        "numpy",
        "ndarray",
        "dtype",
        "i4",
        "f8",
    }
    if not required.issubset(strings):
        raise ValueError("pickle is not the supported RLMRec SciPy COO representation")
    if sum(opcode.name == "STACK_GLOBAL" for opcode, _, _ in operations) != 4:
        raise ValueError("pickle has an unexpected number of global references")

    shape: tuple[int, int] | None = None
    field_indices: dict[str, int] = {}
    for index, (opcode, argument, _) in enumerate(operations):
        if opcode.name == "SHORT_BINUNICODE" and argument in {
            "row",
            "col",
            "coords",
            "data",
        }:
            field_indices[str(argument)] = index + 1
        if opcode.name == "TUPLE2" and index >= 2:
            left_op, left, _ = operations[index - 2]
            right_op, right, _ = operations[index - 1]
            if left_op.name in _INTEGER_OPCODES and right_op.name in _INTEGER_OPCODES:
                shape = (int(left), int(right))
    if shape is None or min(shape) <= 0:
        raise ValueError("pickle does not contain a valid two-dimensional COO shape")
    legacy_fields = {"row", "col", "data"}
    modern_fields = {"coords", "data"}
    if set(field_indices) == legacy_fields:
        row_buffer = _single_aligned_buffer(
            operations,
            field_indices["row"],
            field_indices["col"] - 1,
            alignment=4,
        )
        col_buffer = _single_aligned_buffer(
            operations,
            field_indices["col"],
            field_indices["data"] - 1,
            alignment=4,
        )
    elif set(field_indices) == modern_fields:
        coordinate_buffers = [
            bytes(argument)
            for opcode, argument, _ in operations[
                field_indices["coords"] : field_indices["data"] - 1
            ]
            if opcode.name in {"BINBYTES", "SHORT_BINBYTES"}
            and isinstance(argument, bytes)
            and len(argument) % 4 == 0
        ]
        if len(coordinate_buffers) != 2:
            raise ValueError("COO coords must contain exactly two integer buffers")
        row_buffer, col_buffer = coordinate_buffers
    else:
        raise ValueError("pickle must contain one supported COO coordinate representation")
    data_buffer = _next_buffer(operations, field_indices["data"])
    if len(row_buffer) != len(col_buffer) or len(row_buffer) % 4:
        raise ValueError("COO row and column buffers are inconsistent")
    nnz = len(row_buffer) // 4
    if len(data_buffer) != nnz * 8:
        raise ValueError("COO data buffer is inconsistent with the index buffers")

    row = np.frombuffer(row_buffer, dtype="<i4").copy()
    col = np.frombuffer(col_buffer, dtype="<i4").copy()
    data = np.frombuffer(data_buffer, dtype="<f8").copy()
    if nnz == 0 or np.any(row < 0) or np.any(row >= shape[0]):
        raise ValueError("COO row indices are outside the declared shape")
    if np.any(col < 0) or np.any(col >= shape[1]):
        raise ValueError("COO column indices are outside the declared shape")
    if not np.isfinite(data).all():
        raise ValueError("COO data contains non-finite values")
    return coo_matrix((data, (row, col)), shape=shape)
