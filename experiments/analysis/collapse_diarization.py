import argparse
import json
from pathlib import Path


def build_default_output_path(input_json_path: Path) -> str:
    stem = input_json_path.stem.replace("_02_diarization", "")
    return f"{stem}_collapse.json"


def load_document(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collapse_segments(segments):
    collapsed = []

    current_turn = None

    for s in segments:
        speaker_id = s.get("speaker_id")
        start = s.get("time", {}).get("start_seconds")
        end = s.get("time", {}).get("end_seconds")
        seg_id = s.get("segment_id")

        if current_turn is None:
            current_turn = {
                "speaker_id": speaker_id,
                "start": start,
                "end": end,
                "segment_ids": [seg_id],
            }
            continue

        if speaker_id == current_turn["speaker_id"]:
            # extend current turn
            current_turn["end"] = end
            current_turn["segment_ids"].append(seg_id)
        else:
            # close current turn
            collapsed.append(current_turn)

            # start new one
            current_turn = {
                "speaker_id": speaker_id,
                "start": start,
                "end": end,
                "segment_ids": [seg_id],
            }

    # don't forget last one
    if current_turn:
        collapsed.append(current_turn)

    return collapsed


def scan(input_path: Path, output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = input_path.with_name("collapsed_turns.json")

    document = load_document(input_path)
    segments = document.get("diarization", {}).get("raw_segments", [])

    collapsed = collapse_segments(segments)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(collapsed, f, ensure_ascii=False, indent=2)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="regex match official formula into a CSV."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to the input JSON file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = scan(input_path=args.input_json, output_path=args.output)
    print(f"Regex extraction CSV written to: {output_path}")


if __name__ == "__main__":
    main()
