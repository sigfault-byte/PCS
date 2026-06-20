from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assemblybot.models.flags import SegmentFlag, flags_to_list


def dot_product(vector_blob: bytes, query_blob: bytes) -> float:
    """SQLite UDF: dot product between two float32 vector BLOBs."""
    vector = np.frombuffer(vector_blob, dtype=np.float32)
    query = np.frombuffer(query_blob, dtype=np.float32)

    if vector.shape != query.shape:
        raise ValueError(
            f"Vector shape mismatch in dot_product: {vector.shape} != {query.shape}"
        )

    return float(vector @ query)


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
