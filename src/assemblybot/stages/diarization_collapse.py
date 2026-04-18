import argparse
import json
from pathlib import Path

from assemblybot.models.time import TimeRange, now_utc_iso


def load_document(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_document(doc: dict, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def build_default_collapse_output_path(input_json_path: Path) -> Path:
    stem = input_json_path.stem.replace("_02_diarization", "")
    return input_json_path.with_name(f"{stem}_03_diarization_collapsed.json")


def collapse_diarization_segments(
    raw_segments: list[dict],
) -> list[dict]:
    """
    Collapse consecutive diarization raw segments when speaker_id stays the same.

    This is intentionally simple:
    - no gap threshold yet
    - no embedding logic yet
    - no overlap logic yet

    Result:
        one collapsed segment = one consecutive speaker turn
    """
    if not raw_segments:
        return []

    raw_segments_sorted = sorted(
        raw_segments,
        key=lambda seg: seg["time"]["start_seconds"],
    )

    collapsed_segments: list[dict] = []
    current = None
    collapsed_idx = 0

    for seg in raw_segments_sorted:
        speaker_id = seg["speaker_id"]
        start_seconds = seg["time"]["start_seconds"]
        end_seconds = seg["time"]["end_seconds"]
        source_segment_id = seg["segment_id"]

        # --------------------------------------------------------------
        # Start the first collapsed turn.
        # --------------------------------------------------------------
        if current is None:
            collapsed_idx += 1
            current = {
                "segment_id": f"cdia_{collapsed_idx:06d}",
                "time": {
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "duration_seconds": 0.0,  # recomputed when flushed
                    "start_ts": TimeRange.from_seconds(
                        start_seconds, end_seconds
                    ).start_ts,
                    "end_ts": TimeRange.from_seconds(start_seconds, end_seconds).end_ts,
                },
                "speaker_id": speaker_id,
                "source_diarization_segment_ids": [source_segment_id],
                "segments_count": 1,
            }
            continue

        # --------------------------------------------------------------
        # Same speaker => extend current turn.
        # --------------------------------------------------------------
        if speaker_id == current["speaker_id"]:
            current["time"]["end_seconds"] = end_seconds
            current["time"]["end_ts"] = TimeRange.from_seconds(
                current["time"]["start_seconds"],
                end_seconds,
            ).end_ts
            current["source_diarization_segment_ids"].append(source_segment_id)
            current["segments_count"] += 1
            continue

        # --------------------------------------------------------------
        # Speaker changed => flush previous turn, start a new one.
        # --------------------------------------------------------------
        current["time"]["duration_seconds"] = (
            current["time"]["end_seconds"] - current["time"]["start_seconds"]
        )
        collapsed_segments.append(current)

        collapsed_idx += 1
        time_range = TimeRange.from_seconds(start_seconds, end_seconds)
        current = {
            "segment_id": f"cdia_{collapsed_idx:06d}",
            "time": {
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "duration_seconds": 0.0,
                "start_ts": time_range.start_ts,
                "end_ts": time_range.end_ts,
            },
            "speaker_id": speaker_id,
            "source_diarization_segment_ids": [source_segment_id],
            "segments_count": 1,
        }

    # --------------------------------------------------------------
    # Flush the final collapsed turn.
    # --------------------------------------------------------------
    if current is not None:
        current["time"]["duration_seconds"] = (
            current["time"]["end_seconds"] - current["time"]["start_seconds"]
        )
        collapsed_segments.append(current)

    return collapsed_segments


def collapse_diarization_in_document(
    input_json_path: Path,
    output_json_path: Path | None = None,
) -> dict:
    """
    Transitional stage:
    read raw diarization from canonical JSON,
    collapse consecutive same-speaker diarization segments,
    write collapsed result back into the JSON.
    """
    input_json_path = input_json_path.resolve()

    if not input_json_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

    output_json_path = output_json_path or build_default_collapse_output_path(
        input_json_path
    )

    doc = load_document(input_json_path)

    raw_segments = doc.get("diarization", {}).get("raw_segments", [])
    collapsed_segments = collapse_diarization_segments(raw_segments)

    doc["diarization"]["collapsed_segments"] = collapsed_segments
    doc["diarization"]["collapsed_segments_count"] = len(collapsed_segments)

    doc["pipeline"].setdefault("stage_outputs", {})
    doc["pipeline"]["stage_outputs"]["diarization_collapse"] = str(output_json_path)

    doc["pipeline"]["updated_at"] = now_utc_iso()

    save_document(doc, output_json_path)

    print(f"Raw diarization segments: {len(raw_segments)}")
    print(f"Collapsed diarization segments: {len(collapsed_segments)}")
    print(f"Output: {output_json_path}")

    return doc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collapse consecutive diarization segments by speaker."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to canonical JSON with diarization.raw_segments",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collapse_diarization_in_document(
        input_json_path=args.input_json,
        output_json_path=args.output_json,
    )


if __name__ == "__main__":
    main()
