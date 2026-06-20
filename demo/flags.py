from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assemblybot.models.flags import SegmentFlag

# uv run python demo/flags.py --db data/runs/YOUR_RUN/sqlite/assemblybot.sqlite --table transcript_segment
FLAG_DESCRIPTIONS: dict[SegmentFlag, str] = {
    SegmentFlag.IMPOSSIBLE_SPEECH_RATE: ("Text is too dense for the segment duration."),
    SegmentFlag.INFORMATION_RATE_TOO_HIGH: (
        "UTF-8 byte rate is too high, often from repeated or dense transcript text."
    ),
    SegmentFlag.NEEDS_REVIEW: "Manual or downstream marker for inspection.",
    SegmentFlag.GIBBERISH: "Transcript text appears nonsensical.",
    SegmentFlag.NON_FRENCH: "Text was detected as not French.",
    SegmentFlag.MOSTLY_SILENCE_WITH_SHORT_EVENT: (
        "Mostly quiet audio with a short event that may have triggered hallucinated speech."
    ),
    SegmentFlag.ADJACENT_INFORMATION_RATE_ANOMALY: (
        "Neighbor of a segment with an information-rate anomaly."
    ),
    SegmentFlag.OUTSIDE_VAD: "Whisper segment has no overlap with VAD speech.",
    SegmentFlag.PARTIAL_VAD_OVERLAP: (
        "Whisper segment overlaps VAD, but coverage is below threshold."
    ),
    SegmentFlag.DISCONTIGUOUS_VAD_COVERAGE: (
        "VAD coverage inside the segment has a long internal gap."
    ),
    SegmentFlag.LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE: (
        "Long Whisper segment has too little total VAD coverage."
    ),
    SegmentFlag.LOW_WHISPER_CONFIDENCE: (
        "Whisper average log probability is below the confidence floor."
    ),
    SegmentFlag.HIGH_NO_SPEECH_PROB: (
        "Whisper no-speech probability is above threshold."
    ),
    SegmentFlag.HIGH_COMPRESSION_RATIO: (
        "Whisper compression ratio is high, often a repetitive decoder artifact."
    ),
    SegmentFlag.LONG_DURATION_SHORT_TEXT: (
        "Segment duration is long, but text has too few words or characters."
    ),
    SegmentFlag.DIARIZATION_OVERLAP: (
        "Diarization segment intersects a diarization overlap region."
    ),
    SegmentFlag.MULTI_SPEAKER_CANDIDATE: ("Segment may contain more than one speaker."),
    SegmentFlag.SPEAKER_CHANGE_NEARBY: (
        "Segment is close to a speaker boundary where attribution may be unstable."
    ),
    SegmentFlag.TIE_BREAK_SPEAKER: (
        "Speaker assignment was decided by deterministic tie break."
    ),
    SegmentFlag.ORPHAN_TRANSCRIPT: (
        "Transcript content could not be matched to diarization."
    ),
    SegmentFlag.ORPHAN_DIARIZATION: (
        "Diarization speech could not be matched to transcript content."
    ),
    SegmentFlag.UNSAFE_FOR_SPEAKER_CENTROID: (
        "Diarization segment is linked to a hallucination or noisy transcript."
    ),
}


def active_flags(value: int) -> list[SegmentFlag]:
    return [
        flag for flag in SegmentFlag if flag is not SegmentFlag.NONE and flag & value
    ]


def known_mask() -> int:
    mask = 0
    for flag in SegmentFlag:
        mask |= int(flag)
    return mask


def unknown_bits(value: int) -> int:
    return value & ~known_mask()


def decode_flags(value: int) -> dict[str, object]:
    flags = active_flags(value)
    return {
        "value": value,
        "flags": [
            {
                "name": flag.name,
                "bit": flag.bit_length() - 1,
                "value": int(flag),
                "description": FLAG_DESCRIPTIONS.get(flag, ""),
            }
            for flag in flags
        ],
        "unknown_bits": unknown_bits(value),
    }


def parse_int(value: str) -> int:
    try:
        return int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not an integer; decimal and 0x-prefixed hex are supported"
        ) from error


def print_text(decoded: dict[str, object]) -> None:
    value = int(decoded["value"])
    flags = decoded["flags"]
    unknown = int(decoded["unknown_bits"])

    print(f"{value}:")
    if not flags and unknown == 0:
        print("  NONE")
        return

    for flag in flags:
        item = dict(flag)
        print(
            f"  - {item['name']} "
            f"(bit {item['bit']}, value {item['value']}): {item['description']}"
        )

    if unknown:
        print(f"  - UNKNOWN_BITS: {unknown} ({unknown:#x})")


def print_csv(decoded_values: list[dict[str, object]]) -> None:
    print("input,flag_name,bit,flag_value,description")
    for decoded in decoded_values:
        value = decoded["value"]
        flags = decoded["flags"]
        if not flags and int(decoded["unknown_bits"]) == 0:
            print(f"{value},NONE,,,")
        for flag in flags:
            item = dict(flag)
            print(
                f"{value},{item['name']},{item['bit']},{item['value']},"
                f"{json.dumps(item['description'])}"
            )
        unknown = int(decoded["unknown_bits"])
        if unknown:
            print(f"{value},UNKNOWN_BITS,,{unknown},{json.dumps(hex(unknown))}")


def print_db_summary(db_path: Path, table: str) -> None:
    if not db_path.exists():
        raise SystemExit(f"SQLite file not found: {db_path}")

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            f'''
            select flags, count(*) as count
            from "{table}"
            group by flags
            order by count desc
            '''
        ).fetchall()

    print(f"flags,count,meanings")
    for flags, count in rows:
        names = [flag.name for flag in active_flags(int(flags))]
        unknown = unknown_bits(int(flags))
        if unknown:
            names.append(f"UNKNOWN_BITS({unknown:#x})")
        print(f"{flags},{count},{json.dumps(names)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode AssemblyBot SegmentFlag integer bitmasks."
    )
    parser.add_argument(
        "values",
        nargs="*",
        help=(
            "Flag integer(s) to decode, or a SQLite DB path. "
            "Integers may be decimal or 0x-prefixed hex."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all known flags and their bit values.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "csv"),
        default="text",
        help="Output format for decoded values.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="SQLite DB path. Prints grouped flag counts from the selected table.",
    )
    parser.add_argument(
        "--table",
        default="turn",
        help='Table to summarize when using --db. Default: "turn".',
    )
    parser.add_argument(
        "--table-turn",
        action="store_const",
        const="turn",
        dest="table",
        help='Shortcut for --table "turn".',
    )
    parser.add_argument(
        "--table-transcript",
        action="store_const",
        const="transcript_segment",
        dest="table",
        help='Shortcut for --table "transcript_segment".',
    )
    parser.add_argument(
        "--table-diarization",
        action="store_const",
        const="diarization_segment",
        dest="table",
        help='Shortcut for --table "diarization_segment".',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    positional_db_paths = [
        Path(value)
        for value in args.values
        if value.endswith((".sqlite", ".sqlite3", ".db"))
    ]

    if args.db or positional_db_paths:
        if args.db and positional_db_paths:
            parser.error("provide the SQLite path either positionally or with --db, not both")
        print_db_summary(args.db or positional_db_paths[0], args.table)
        return

    if args.list:
        decoded_values = [
            decode_flags(int(flag)) for flag in active_flags(known_mask())
        ]
    else:
        if not args.values:
            parser.error("provide at least one flag integer, or use --list")
        decoded_values = [decode_flags(parse_int(value)) for value in args.values]

    if args.format == "json":
        print(json.dumps(decoded_values, indent=2))
    elif args.format == "csv":
        print_csv(decoded_values)
    else:
        for index, decoded in enumerate(decoded_values):
            if index:
                print()
            print_text(decoded)


if __name__ == "__main__":
    main()
