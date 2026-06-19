from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from sqlite_function import register_sqlite_functions


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db" / "assemblybot.sqlite"
MODEL_NAME = "h4c5/sts-camembert-base"
TOP_K = 5


SEARCH_SQL = """
select
    dot_product(e.vector, ?) as score,
    sc.turn_id,
    sc.chunk_index,
    p.name as speaker_name,
    p.role as speaker_role,
    sc.text as chunk_text
from semantic_chunk sc
join embedding e on e.id = sc.embedding_id
left join turn_analysis ta on ta.turn_id = sc.turn_id
left join person p on p.id = ta.current_person_id
order by score desc
limit ?
"""


def embed_query(model: SentenceTransformer, query: str) -> bytes:
    embedding = model.encode([query], normalize_embeddings=True)[0]
    return np.asarray(embedding, dtype=np.float32).tobytes()


def print_result(
    *,
    rank: int,
    score: float,
    turn_id: int,
    chunk_index: int,
    speaker_name: str | None,
    speaker_role: str | None,
    chunk_text: str,
) -> None:
    print("=" * 80)
    print(f"#{rank} score={score:.4f} turn_id={turn_id} chunk_index={chunk_index}")

    if speaker_name:
        role = f" ({speaker_role})" if speaker_role else ""
        print(f"speaker={speaker_name}{role}")
    else:
        print("speaker=<unknown>")

    print(chunk_text)


def main() -> None:
    query = input("Enter query: ").strip()
    if not query:
        print("No query provided.")
        return

    model = SentenceTransformer(MODEL_NAME)
    query_blob = embed_query(model, query)

    with sqlite3.connect(DB_PATH) as connection:
        register_sqlite_functions(connection)
        rows = connection.execute(SEARCH_SQL, (query_blob, TOP_K)).fetchall()

    for rank, row in enumerate(rows, start=1):
        print_result(
            rank=rank,
            score=float(row[0]),
            turn_id=int(row[1]),
            chunk_index=int(row[2]),
            speaker_name=row[3],
            speaker_role=row[4],
            chunk_text=row[5],
        )


if __name__ == "__main__":
    main()
