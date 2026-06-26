from __future__ import annotations

import argparse
import sqlite3
import textwrap
from pathlib import Path

import numpy as np
from demo_queries_fand_flags import QUERIES, SegmentFlag
from sentence_transformers import SentenceTransformer
from sqlite_function import register_sqlite_functions

DB_PATH = Path(__file__).resolve().with_name("assemblybot.sqlite")
MODEL_NAME = "h4c5/sts-camembert-base"
TOP_K = 5

LIST_SQL_TABLE = """
select 'person' as table_name, count(*) as rows from person
union all
select 'speaker_cluster', count(*) from speaker_cluster
union all
select 'transcript_segment', count(*) from transcript_segment
union all
select 'diarization_segment', count(*) from diarization_segment
union all
select 'turn', count(*) from turn
union all
select 'turn_analysis', count(*) from turn_analysis
union all
select 'semantic_chunk', count(*) from semantic_chunk
union all
select 'embedding', count(*) from embedding;
"""

SEARCH_SQL = """
select
    dot_product(e.vector, ?) as score,
    sc.turn_id,
    sc.chunk_index,
    t.start_seconds,
    s.slug as session_slug,
    p.name as speaker_name,
    p.role as speaker_role,
    flag_names(t.flags) as flags,
    sc.text as chunk_text
from semantic_chunk sc
join embedding e on e.id = sc.embedding_id
left join turn_analysis ta on ta.turn_id = sc.turn_id
left join person p on p.id = ta.current_person_id
join turn as t on t.id = sc.turn_id
join session as s on t.session_id = s.id
order by score desc
limit ?
"""

TURN_SQL = """
select
    t.id as turn_id,
    t.start_seconds,
    t.end_seconds,
    t.flags,
    s.slug as session_slug,
    p.name as speaker_name,
    p.role as speaker_role,
    t.text
from turn as t
left join turn_analysis ta on ta.turn_id = t.id
left join person p on p.id = ta.current_person_id
join session as s on t.session_id = s.id
where t.id = ?
"""

FLAG_DESCRIPTIONS = {
    SegmentFlag.IMPOSSIBLE_SPEECH_RATE: "Text is too dense for the segment duration.",
    SegmentFlag.INFORMATION_RATE_TOO_HIGH: "UTF-8 byte rate is unusually high.",
    SegmentFlag.NEEDS_REVIEW: "This turn was marked for human review.",
    SegmentFlag.GIBBERISH: "Transcript text may be nonsensical.",
    SegmentFlag.NON_FRENCH: "Text may not be French.",
    SegmentFlag.MOSTLY_SILENCE_WITH_SHORT_EVENT: (
        "Mostly quiet audio with a short event."
    ),
    SegmentFlag.ADJACENT_INFORMATION_RATE_ANOMALY: (
        "Near a suspicious dense-text segment."
    ),
    SegmentFlag.OUTSIDE_VAD: "No overlap with voice activity detection speech.",
    SegmentFlag.PARTIAL_VAD_OVERLAP: "Only partial overlap with detected speech.",
    SegmentFlag.DISCONTIGUOUS_VAD_COVERAGE: (
        "Detected speech has a long internal gap."
    ),
    SegmentFlag.LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE: (
        "Long segment with little detected speech."
    ),
    SegmentFlag.LOW_WHISPER_CONFIDENCE: "Whisper confidence is low.",
    SegmentFlag.HIGH_NO_SPEECH_PROB: "Whisper thought this may be non-speech.",
    SegmentFlag.HIGH_COMPRESSION_RATIO: ("Text may be repetitive or decoder noise."),
    SegmentFlag.LONG_DURATION_SHORT_TEXT: (
        "Long audio duration with very little text."
    ),
    SegmentFlag.DIARIZATION_OVERLAP: (
        "Speaker diarization detected overlapping speech."
    ),
    SegmentFlag.MULTI_SPEAKER_CANDIDATE: (
        "This turn may contain more than one speaker."
    ),
    SegmentFlag.SPEAKER_CHANGE_NEARBY: "Close to a speaker boundary.",
    SegmentFlag.TIE_BREAK_SPEAKER: "Speaker was assigned by a tie-break rule.",
    SegmentFlag.ORPHAN_TRANSCRIPT: ("Transcript could not be matched to diarization."),
    SegmentFlag.ORPHAN_DIARIZATION: (
        "Diarization speech could not be matched to transcript."
    ),
    SegmentFlag.UNSAFE_FOR_SPEAKER_CENTROID: (
        "Linked to noisy or hallucinated transcript audio."
    ),
}


def embed_query(model: SentenceTransformer, query: str) -> bytes:
    embedding = model.encode([query], normalize_embeddings=True)[0]
    return np.asarray(embedding, dtype=np.float32).tobytes()


def format_time(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def print_table(headers: list[str], rows: list[tuple[object, ...]]) -> None:
    if not rows:
        print("No results found.")
        return

    widths = [
        max(len(str(x)) for x in [header] + [row[i] for row in rows])
        for i, header in enumerate(headers)
    ]

    sep = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    print(sep)
    print(
        "| "
        + " | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers))
        + " |"
    )
    print(sep)

    for row in rows:
        print(
            "| "
            + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
            + " |"
        )

    print(sep)


def active_flags(flags: int) -> list[SegmentFlag]:
    return [
        flag
        for flag in SegmentFlag
        if flag is not SegmentFlag.NONE and bool(flag & flags)
    ]


def choose_query() -> str:
    print("Choose a query, or type your own search text:\n")
    for index, (label, _vector) in enumerate(QUERIES, start=1):
        print(f"\t{index}) {label}")
    print()

    try:
        choice = input("Enter choice or query: ").strip()
    except EOFError:
        return ""
    if not choice:
        return ""

    try:
        query_index = int(choice)
    except ValueError:
        return choice

    if 1 <= query_index <= len(QUERIES):
        return QUERIES[query_index - 1][0]

    print("That number is not one of the demo queries, so I will search it as text.")
    return choice


def print_turn_detail(row: sqlite3.Row) -> None:
    speaker = row["speaker_name"] or "<unknown>"
    if row["speaker_role"]:
        speaker = f"{speaker} ({row['speaker_role']})"

    print("\n" + "=" * 80)
    print(f"Turn {row['turn_id']}")
    print(f"File: \n\t{row['session_slug']}")
    print(
        f"Time: \n\t{format_time(row['start_seconds'])} - "
        f"{format_time(row['end_seconds'])}"
    )
    print(f"Speaker: \n\t{speaker}")

    print("\nFull text:\n")
    print(textwrap.fill(row["text"], width=100))

    flags = active_flags(int(row["flags"]))
    print("\nFlags:\n")
    if not flags:
        print("  NONE: no quality flags attached to this turn.")
    else:
        for flag in flags:
            description = FLAG_DESCRIPTIONS.get(flag, "No description available.")
            print(f"  - {flag.name}: {description}")
    print("=" * 80)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semantic search and retrieval over the demo AssemblyBot SQLite DB."
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Search text. Omit it for the interactive demo query picker.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help=f"SQLite DB path. Default: {DB_PATH}",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"SentenceTransformer model name. Default: {MODEL_NAME}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of results to return. Default: {TOP_K}",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    db_path = args.db

    if not db_path.exists():
        print(f"SQLite database not found: {db_path}")
        return

    query = " ".join(args.query).strip() if args.query else choose_query()
    if not query:
        print("No query provided.")
        return

    print("\n")
    print("=" * 50)
    print("\tAssemblyBot Semantic Search")
    print("=" * 50)
    print(f"\nQuery:\n\t{query}\n")
    print(f"Embedding query with {args.model}...")

    model = SentenceTransformer(args.model)
    query_blob = embed_query(model, query)

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        register_sqlite_functions(connection)
        overview_rows = connection.execute(LIST_SQL_TABLE).fetchall()
        rows = connection.execute(SEARCH_SQL, (query_blob, args.top_k)).fetchall()

    print("\nDatabase overview")
    print_table(
        ["table name", "number of rows"],
        [tuple(row) for row in overview_rows],
    )
    print()

    table_rows = [
        (
            rank,
            row["turn_id"],
            f"{row['score']:.4f}",
            format_time(row["start_seconds"]),
            textwrap.shorten(row["session_slug"], width=48, placeholder="..."),
            row["speaker_name"] or "<unknown>",
            textwrap.shorten(row["chunk_text"], width=90, placeholder="..."),
        )
        for rank, row in enumerate(rows, start=1)
    ]
    print_table(
        ["#", "turn_id", "score", "time", "file", "speaker", "excerpt"],
        table_rows,  # type: ignore
    )

    returned_turn_ids = {int(row["turn_id"]) for row in rows}
    try:
        selected = input(
            "\nEnter a turn_id from the table to read it, or press Enter to quit: "
        ).strip()
        selected_turn_id = int(selected) if selected else None
    except (EOFError, ValueError):
        return

    if selected_turn_id is None:
        return

    if selected_turn_id not in returned_turn_ids:
        print("Please choose a turn_id from the current search results.")
        return

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        detail = connection.execute(TURN_SQL, (selected_turn_id,)).fetchone()

    if detail is None:
        print("Turn not found.")
        return

    print_turn_detail(detail)


if __name__ == "__main__":
    main()
