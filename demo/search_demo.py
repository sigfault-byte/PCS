import array
import sqlite3
import textwrap
from pathlib import Path

from demo_queries_fand_flags import QUERIES, SegmentFlag
from sqlite_function import register_sqlite_functions

DB_PATH = Path(__file__).resolve().with_name("assemblybot.sqlite")
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
    "IMPOSSIBLE_SPEECH_RATE": "Text is too dense for the segment duration.",
    "INFORMATION_RATE_TOO_HIGH": "UTF-8 byte rate is unusually high.",
    "NEEDS_REVIEW": "This turn was marked for human review.",
    "GIBBERISH": "Transcript text may be nonsensical.",
    "NON_FRENCH": "Text may not be French.",
    "MOSTLY_SILENCE_WITH_SHORT_EVENT": "Mostly quiet audio with a short event.",
    "ADJACENT_INFORMATION_RATE_ANOMALY": "Near a suspicious dense-text segment.",
    "OUTSIDE_VAD": "No overlap with voice activity detection speech.",
    "PARTIAL_VAD_OVERLAP": "Only partial overlap with detected speech.",
    "DISCONTIGUOUS_VAD_COVERAGE": "Detected speech has a long internal gap.",
    "LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE": "Long segment with little detected speech.",
    "LOW_WHISPER_CONFIDENCE": "Whisper confidence is low.",
    "HIGH_NO_SPEECH_PROB": "Whisper thought this may be non-speech.",
    "HIGH_COMPRESSION_RATIO": "Text may be repetitive or decoder noise.",
    "LONG_DURATION_SHORT_TEXT": "Long audio duration with very little text.",
    "DIARIZATION_OVERLAP": "Speaker diarization detected overlapping speech.",
    "MULTI_SPEAKER_CANDIDATE": "This turn may contain more than one speaker.",
    "SPEAKER_CHANGE_NEARBY": "Close to a speaker boundary.",
    "TIE_BREAK_SPEAKER": "Speaker was assigned by a tie-break rule.",
    "ORPHAN_TRANSCRIPT": "Transcript could not be matched to diarization.",
    "ORPHAN_DIARIZATION": "Diarization speech could not be matched to transcript.",
    "UNSAFE_FOR_SPEAKER_CENTROID": "Linked to noisy or hallucinated transcript audio.",
}


def vector_to_blob(vector: list[float]) -> bytes:
    query = array.array("f", vector)
    return query.tobytes()


def format_time(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def print_table(headers, rows):
    if not rows:
        print("No results found.")
        return

    widths = [
        max(len(str(x)) for x in [header] + [row[i] for row in rows])
        for i, header in enumerate(headers)
    ]

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    print(sep)
    print(
        "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    )
    print(sep)

    for row in rows:
        print(
            "| "
            + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
            + " |"
        )

    print(sep)


def active_flag_names(flags: int) -> list[str]:
    return [
        flag.name
        for flag in SegmentFlag
        if flag is not SegmentFlag.NONE and bool(flag & flags)
    ]  # type: ignore


def print_tables(row: sqlite3.Row) -> None:
    pass


def print_turn_detail(row: sqlite3.Row) -> None:
    speaker = row["speaker_name"] or "<unknown>"
    if row["speaker_role"]:
        speaker = f"{speaker} ({row['speaker_role']})"

    print("\n" + "=" * 80)
    print(f"Turn {row['turn_id']}")
    print(f"File: \n\t{row['session_slug']}")
    print(
        f"Time: \n\t{format_time(row['start_seconds'])} - {format_time(row['end_seconds'])}"
    )
    print(f"Speaker: \n\t{speaker}")

    print("\nFull text:\n")
    print(textwrap.fill(row["text"], width=100))

    flag_names = active_flag_names(int(row["flags"]))
    print("\nFlags:\n")
    if not flag_names:
        print("  NONE: no quality flags attached to this turn.")
    else:
        for name in flag_names:
            description = FLAG_DESCRIPTIONS.get(name, "No description available.")
            print(f"  - {name}: {description}")
    print("=" * 80)


def main():
    if not DB_PATH.exists():
        print(f"SQLite demo database not found: {DB_PATH}")
        return

    print("\n")
    print("=" * 50)
    print("\tAssemblyBot Semantic Search Demo")
    print("=" * 50)

    print("\n")

    print("Choose a query:\n ")
    for index, (label, _vector) in enumerate(QUERIES, start=1):
        print(f"\t{index}) {label}")
    print("\n")

    try:
        choice = int(input("Enter choice: ").strip())
    except ValueError:
        print("Choose one of the numbered queries.")
        return

    if choice < 1 or choice > len(QUERIES):
        print("Choose one of the numbered queries.")
        return

    query_text, query_vector = QUERIES[choice - 1]
    query_blob = vector_to_blob(query_vector)

    print(f"\nQuery:\n\t{query_text}\n")

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        register_sqlite_functions(connection)
        table_rows = connection.execute(LIST_SQL_TABLE).fetchall()
        rows = connection.execute(SEARCH_SQL, (query_blob, TOP_K)).fetchall()

    print("\n")
    print("Database overview")
    print_table(["table name", "number of rows"], table_rows)
    print("\n")

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
        table_rows,
    )

    returned_turn_ids = {int(row["turn_id"]) for row in rows}
    try:
        selected_turn_id = int(
            input(
                "\nEnter a turn_id from the table to read it, or press Enter to quit: "
            ).strip()
        )
    except ValueError:
        return

    if selected_turn_id not in returned_turn_ids:
        print("Please choose a turn_id from the current search results.")
        return

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        detail = connection.execute(TURN_SQL, (selected_turn_id,)).fetchone()

    if detail is None:
        print("Turn not found.")
        return

    print_turn_detail(detail)


if __name__ == "__main__":
    main()
