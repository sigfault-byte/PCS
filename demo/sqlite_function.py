from __future__ import annotations

import json
import math
import sqlite3
from array import array

from demo_queries_fand_flags import SegmentFlag, flags_to_list


def float32_array_from_blob(blob: bytes) -> array:
    """Decode a SQLite BLOB containing native float32 values."""
    vector = array("f")
    if len(blob) % vector.itemsize:
        raise ValueError(
            f"Float32 vector BLOB length must be a multiple of {vector.itemsize}: "
            f"got {len(blob)} bytes"
        )

    vector.frombytes(blob)
    return vector


def dot_product(vector_blob: bytes, query_blob: bytes) -> float:
    """SQLite UDF: dot product between two float32 vector BLOBs."""
    vector = float32_array_from_blob(vector_blob)
    query = float32_array_from_blob(query_blob)

    if len(vector) != len(query):
        raise ValueError(
            f"Vector length mismatch in dot_product: {len(vector)} != {len(query)}"
        )

    return math.fsum(left * right for left, right in zip(vector, query))


def flag_names(flags: int) -> str:
    """SQLite UDF: JSON array of active SegmentFlag names."""
    return json.dumps(flags_to_list(SegmentFlag(flags)))


def has_flag_name(flags: int, flag_name: str) -> int:
    """SQLite UDF: 1 when a SegmentFlag bit is present, otherwise 0."""
    try:
        flag = SegmentFlag[flag_name]
    except KeyError:
        return 0

    return int(bool(SegmentFlag(flags) & flag))


def register_sqlite_functions(connection: sqlite3.Connection) -> None:
    """Register demo SQLite helper functions."""
    connection.create_function("dot_product", 2, dot_product)
    connection.create_function("flag_names", 1, flag_names)
    connection.create_function("has_flag_name", 2, has_flag_name)
