from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlite_function import register_sqlite_functions

# data/runs/2026-06-20_2026-06-09_1ere-seance-questions-au-gouvernement-conventions-france-finlande-et-france-suede-en-matiere-d-i_342c2d13/sqlite/assemblybot.sqlite
DB_PATH = Path(__file__).resolve().with_name("assemblybot.sqlite")

MODEL_NAME = "h4c5/sts-camembert-base"
TOP_K = 5


SEARCH_SQL = """
select
    dot_product(e.vector, ?) as score,
    sc.turn_id,
    sc.chunk_index,
    t.start_seconds,
    p.name as speaker_name,
    p.role as speaker_role,
    sc.text as chunk_text
from semantic_chunk sc
join embedding e on e.id = sc.embedding_id
left join turn_analysis ta on ta.turn_id = sc.turn_id
left join person p on p.id = ta.current_person_id
join turn as t on t.id = sc.turn_id
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
    start_second: float,
    chunk_index: int,
    speaker_name: str | None,
    speaker_role: str | None,
    chunk_text: str,
) -> None:
    print("=" * 80)
    print(f"#{rank} score={score:.4f} turn_id={turn_id} audio timestamp={start_second} chunk_index={chunk_index}")

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
        connection.row_factory = sqlite3.Row
        register_sqlite_functions(connection)
        rows = connection.execute(SEARCH_SQL, (query_blob, TOP_K)).fetchall()

    for rank, row in enumerate(rows, start=1):
        print_result(
            rank=rank,
            score=float(row["score"]),
            turn_id=int(row["turn_id"]),
            start_second=float(row["start_seconds"]),
            chunk_index=int(row["chunk_index"]),
            speaker_name=row["speaker_name"],
            speaker_role=row["speaker_role"],
            chunk_text=row["chunk_text"],
        )


if __name__ == "__main__":
    main()
