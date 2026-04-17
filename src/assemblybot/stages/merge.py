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


def build_token_to_transcript_segment_map(
    transcript_segments: list[dict],
) -> dict[int, str]:
    """
    Map each token_id to its parent Whisper raw segment id.

    Transcript raw segments are anchors only:
    they reference a contiguous token range via start_token_id/end_token_id.
    """
    token_to_segment: dict[int, str] = {}

    for seg in transcript_segments:
        start_token_id = seg.get("start_token_id")
        end_token_id = seg.get("end_token_id")

        if start_token_id is None or end_token_id is None:
            continue

        if start_token_id < 0 or end_token_id < 0:
            continue

        for token_id in range(start_token_id, end_token_id + 1):
            token_to_segment[token_id] = seg["segment_id"]

    return token_to_segment


def assign_tokens_to_diarization_segments(
    transcript_tokens: list[dict],
    diarization_segments: list[dict],
) -> dict[str, list[dict]]:
    """
    Assign each transcript raw token to a diarization segment
    using token midpoint containment.

    Returns:
        dict[dia_segment_id] -> list of token dicts
    """
    diarization_segments_sorted = sorted(
        diarization_segments,
        key=lambda seg: seg["time"]["start_seconds"],
    )
    transcript_tokens_sorted = sorted(
        transcript_tokens,
        key=lambda tok: tok["start_seconds"],
    )

    assigned: dict[str, list[dict]] = {
        d_seg["segment_id"]: [] for d_seg in diarization_segments_sorted
    }

    dia_idx = 0
    dia_count = len(diarization_segments_sorted)

    for tok in transcript_tokens_sorted:
        tmid = (tok["start_seconds"] + tok["end_seconds"]) / 2.0

        while dia_idx < dia_count:
            current = diarization_segments_sorted[dia_idx]
            current_end = current["time"]["end_seconds"]

            if tmid <= current_end:
                break

            dia_idx += 1

        if dia_idx >= dia_count:
            break

        current = diarization_segments_sorted[dia_idx]
        current_start = current["time"]["start_seconds"]
        current_end = current["time"]["end_seconds"]

        if current_start <= tmid <= current_end:
            assigned[current["segment_id"]].append(tok)

    return assigned


def build_token_provenance(tokens: list[dict]) -> tuple[int | None, int | None]:
    if not tokens:
        return None, None

    return tokens[0]["token_id"], tokens[-1]["token_id"]


def build_transcript_segment_provenance(
    tokens: list[dict],
    token_to_transcript_segment: dict[int, str],
) -> list[str]:
    segment_ids: list[str] = []
    seen: set[str] = set()

    for tok in tokens:
        seg_id = token_to_transcript_segment.get(tok["token_id"])
        if seg_id is None or seg_id in seen:
            continue
        seen.add(seg_id)
        segment_ids.append(seg_id)

    return segment_ids


def reconstruct_text_from_tokens(tokens: list[dict]) -> str:
    """
    Rebuild exact raw transcript text from raw tokens.

    Important:
    raw_token keeps Whisper spacing as emitted,
    so we must use ''.join(...) and not ' '.join(...).
    """
    return "".join(tok["raw_token"] for tok in tokens)


def compute_other_speakers_for_diarization_segment(
    current_diarization_segment: dict,
    all_diarization_segments: list[dict],
    min_other_overlap_seconds: float,
    multiple_speaker_ratio_threshold: float,
) -> list[dict]:
    current_time = current_diarization_segment["time"]
    current_start = current_time["start_seconds"]
    current_end = current_time["end_seconds"]
    current_duration = current_time["duration_seconds"]
    current_id = current_diarization_segment["segment_id"]
    current_speaker_id = current_diarization_segment["speaker_id"]

    overlaps_by_speaker: dict[str, float] = {}

    for other in all_diarization_segments:
        other_id = other["segment_id"]
        if other_id == current_id:
            continue

        other_speaker_id = other["speaker_id"]
        if other_speaker_id == current_speaker_id:
            continue

        other_time = other["time"]
        ov = overlap_seconds(
            current_start,
            current_end,
            other_time["start_seconds"],
            other_time["end_seconds"],
        )

        if ov <= 0:
            continue

        overlaps_by_speaker[other_speaker_id] = (
            overlaps_by_speaker.get(other_speaker_id, 0.0) + ov
        )

    other_speakers: list[dict] = []

    for speaker_id, ov in sorted(
        overlaps_by_speaker.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        overlap_ratio = ov / current_duration if current_duration > 0 else 0.0

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

    return other_speakers


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

    transcript_tokens = doc["transcript"]["raw_tokens"]
    transcript_segments = doc["transcript"]["raw_segments"]
    diarization_segments = doc["diarization"]["raw_segments"]

    # ------------------------------------------------------------------
    # Build helper maps once.
    # ------------------------------------------------------------------
    token_to_transcript_segment = build_token_to_transcript_segment_map(
        transcript_segments
    )

    diarization_tokens = assign_tokens_to_diarization_segments(
        transcript_tokens=transcript_tokens,
        diarization_segments=diarization_segments,
    )

    final_segments: list[dict] = []

    # Keep diarization chronology as the final segment chronology.
    diarization_segments_sorted = sorted(
        diarization_segments,
        key=lambda seg: seg["time"]["start_seconds"],
    )

    for idx, d_seg in enumerate(diarization_segments_sorted, start=1):
        d_time = d_seg["time"]
        d_duration = d_time["duration_seconds"]
        d_segment_id = d_seg["segment_id"]
        d_speaker_id = d_seg["speaker_id"]

        assigned_tokens = diarization_tokens.get(d_segment_id, [])

        reconstructed_text = reconstruct_text_from_tokens(assigned_tokens)
        token_start_id, token_end_id = build_token_provenance(assigned_tokens)
        transcript_segment_ids = build_transcript_segment_provenance(
            assigned_tokens,
            token_to_transcript_segment,
        )

        other_speakers = compute_other_speakers_for_diarization_segment(
            current_diarization_segment=d_seg,
            all_diarization_segments=diarization_segments_sorted,
            min_other_overlap_seconds=min_other_overlap_seconds,
            multiple_speaker_ratio_threshold=multiple_speaker_ratio_threshold,
        )

        # ------------------------------------------------------------------
        # Flags
        # ------------------------------------------------------------------
        flags = SegmentFlag.NONE

        # No text recovered from tokens: likely deserves review.
        if not assigned_tokens or not reconstructed_text.strip():
            flags |= SegmentFlag.NEEDS_REVIEW

        # Preserve the old short-segment warning.
        if d_duration < 1.5:
            flags |= SegmentFlag.IS_SHORT

        # Overlapping competing diarization speakers.
        if other_speakers:
            flags |= SegmentFlag.MULTIPLE_SPEAKERS

        final_segment = {
            "segment_id": f"seg_{idx:06d}",
            "time": d_time,
            "speaker": {
                "speaker_id": d_speaker_id,
                "speaker_label": None,
                "speaker_label_source": None,
                # Since this segment is diarization-native, confidence is not
                # overlap-derived anymore. Keep None for now.
                "confidence": None,
            },
            "text": {
                "raw": reconstructed_text,
                "normalized": None,
                "language": doc["transcript"].get("language_detected"),
            },
            "flags": int(flags),
            "other_speakers": other_speakers,
            "entities": [],
            "keywords": [],
            "provenance": {
                "transcript_segment_ids": transcript_segment_ids,
                "diarization_segment_ids": [d_segment_id],
                "transcript_token_start_id": token_start_id,
                "transcript_token_end_id": token_end_id,
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
