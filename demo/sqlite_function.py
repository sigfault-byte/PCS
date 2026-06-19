from __future__ import annotations

import sqlite3

import numpy as np


def dot_product(vector_blob: bytes, query_blob: bytes) -> float:
    """SQLite UDF: dot product between two float32 vector BLOBs."""
    vector = np.frombuffer(vector_blob, dtype=np.float32)
    query = np.frombuffer(query_blob, dtype=np.float32)

    if vector.shape != query.shape:
        raise ValueError(
            f"Vector shape mismatch in dot_product: {vector.shape} != {query.shape}"
        )

    return float(vector @ query)


def register_sqlite_functions(connection: sqlite3.Connection) -> None:
    """Register demo vector-search functions on a sqlite3 connection."""
    connection.create_function("dot_product", 2, dot_product)
