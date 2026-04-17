from __future__ import annotations

import argparse
import json
from pathlib import Path

from assemblybot.config import INTERIM_DIR
from assemblybot.models.flags import SegmentFlag
from assemblybot.models.time import now_utc_iso


def build_default_output_path(input_json_path: Path) -> Path:
    stem = input_json_path.stem.replace("_02_diarization", "")
    return INTERIM_DIR / f"{stem}_03_merged.json"


def load_document(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_document(doc: dict, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def overlap_seconds(
    a_start: float, a_end: float, b_start: float, b_end: float
) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def merge_segments(
    input_json_path: Path,
    output_json_path: Path | None = None,
    min_other_overlap_seconds: float = 0.5,
    multiple_speaker_ratio_threshold: float = 0.2,
) -> dict:
    input_json_path = input_json_path.resolve()

    if not input_json_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

    output_json_path = output_json_path or build_default_output_path(input_json_path)
    doc = load_document(input_json_path)

    transcript_segments = doc["transcript"]["raw_segments"]
    diarization_segments = doc["diarization"]["raw_segments"]

    final_segments: list[dict] = []

    for idx, t_seg in enumerate(transcript_segments, start=1):
        t_time = t_seg["time"]
        t_start = t_time["start_seconds"]
        t_end = t_time["end_seconds"]
        t_duration = t_time["duration_seconds"]

        overlaps_by_speaker: dict[str, float] = {}
        overlap_details: dict[str, list[str]] = {}

        for d_seg in diarization_segments:
            d_time = d_seg["time"]
            d_start = d_time["start_seconds"]
            d_end = d_time["end_seconds"]

            ov = overlap_seconds(t_start, t_end, d_start, d_end)
            if ov <= 0:
                continue

            speaker_id = d_seg["speaker_id"]
            overlaps_by_speaker[speaker_id] = (
                overlaps_by_speaker.get(speaker_id, 0.0) + ov
            )
            overlap_details.setdefault(speaker_id, []).append(d_seg["segment_id"])

        flags = SegmentFlag.NONE
        primary_speaker_id: str | None = None
        primary_confidence: float | None = None
        other_speakers: list[dict] = []
        provenance_diarization_ids: list[str] = []

        if not overlaps_by_speaker:
            flags |= SegmentFlag.NEEDS_REVIEW
        else:
            ranked = sorted(
                overlaps_by_speaker.items(), key=lambda x: x[1], reverse=True
            )
            primary_speaker_id, primary_overlap = ranked[0]
            primary_confidence = (
                primary_overlap / t_duration if t_duration > 0 else None
            )

            provenance_diarization_ids.extend(
                overlap_details.get(primary_speaker_id, [])
            )

            for speaker_id, ov in ranked[1:]:
                overlap_ratio = ov / t_duration if t_duration > 0 else 0.0
                if (
                    ov >= min_other_overlap_seconds
                    or overlap_ratio >= multiple_speaker_ratio_threshold
                ):
                    other_speakers.append(
                        {
                            "speaker_id": speaker_id,
                            "overlap_seconds": ov,
                            "overlap_ratio": overlap_ratio,
                        }
                    )
                    provenance_diarization_ids.extend(
                        overlap_details.get(speaker_id, [])
                    )

            if other_speakers:
                flags |= SegmentFlag.MULTIPLE_SPEAKERS

        if t_duration < 1.5:
            flags |= SegmentFlag.IS_SHORT

        final_segment = {
            "segment_id": f"seg_{idx:06d}",
            "time": t_seg["time"],
            "speaker": {
                "speaker_id": primary_speaker_id,
                "speaker_label": None,
                "speaker_label_source": None,
                "confidence": primary_confidence,
            },
            "text": {
                "raw": t_seg["text"],
                "normalized": None,
                "language": doc["transcript"].get("language_detected"),
            },
            "flags": int(flags),
            "other_speakers": other_speakers,
            "entities": [],
            "keywords": [],
            "provenance": {
                "transcript_segment_ids": [t_seg["segment_id"]],
                "diarization_segment_ids": sorted(set(provenance_diarization_ids)),
                "stage_created_by": "merge",
            },
        }

        final_segments.append(final_segment)

    doc["segments"] = final_segments

    if "merge" not in doc["pipeline"]["stages_completed"]:
        doc["pipeline"]["stages_completed"].append("merge")

    doc["pipeline"]["updated_at"] = now_utc_iso()
    doc["pipeline"]["stage_outputs"]["merge"] = str(output_json_path)

    save_document(doc, output_json_path)

    print(f"Merged segments: {len(doc['segments'])}")
    print(f"Output: {output_json_path}")

    return doc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge transcript and diarization into final speaker-attributed segments."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to transcript+diarization JSON",
    )
    parser.add_argument(
        "--output-json",
        help="Optional output JSON path",
    )
    parser.add_argument(
        "--min-other-overlap-seconds",
        type=float,
        default=0.5,
        help="Minimum overlap in seconds to keep as other speaker",
    )
    parser.add_argument(
        "--multiple-speaker-ratio-threshold",
        type=float,
        default=0.2,
        help="Minimum overlap ratio to keep as other speaker",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    merge_segments(
        input_json_path=Path(args.input_json),
        output_json_path=Path(args.output_json) if args.output_json else None,
        min_other_overlap_seconds=args.min_other_overlap_seconds,
        multiple_speaker_ratio_threshold=args.multiple_speaker_ratio_threshold,
    )


if __name__ == "__main__":
    main()
