from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from assemblybot.config import INTERIM_DIR
from assemblybot.models.flags import SegmentFlag
from assemblybot.models.time import now_utc_iso

TERMINAL_PUNCTUATION_RE = re.compile(r"""[.!?…]+(?:[\]\)\}"'»”]*)\s*$""")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?…])")
MULTISPACE_RE = re.compile(r"\s+")

# Merge behavior thresholds.
SOFT_MIN_CHARS = 120
SOFT_MIN_WORDS = 20
SOFT_MIN_SOURCE_SEGMENTS = 2
HARD_MAX_CHARS = 800
HARD_MAX_DURATION_SECONDS = 15.0

# Attribution thresholds.
DEFAULT_MIN_OTHER_OVERLAP_SECONDS = 0.2
DEFAULT_MULTIPLE_SPEAKER_RATIO_THRESHOLD = 0.2

# Merge veto: preserve useful Whisper boundaries when the next raw segment
# looks like a possible speaker handoff despite same primary speaker id.
MERGE_VETO_ON_PUNCTUATED_BOUNDARY_WITH_OVERLAP = True
MERGE_VETO_MIN_OVERLAP_SECONDS = 0.40
MERGE_VETO_MIN_OVERLAP_RATIO = 0.12


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


def has_meaningful_other_speaker_overlap(item: dict) -> bool:
    for other in item.get("other_speakers", []):
        if (
            other.get("overlap_seconds", 0.0) >= MERGE_VETO_MIN_OVERLAP_SECONDS
            or other.get("overlap_ratio", 0.0) >= MERGE_VETO_MIN_OVERLAP_RATIO
        ):
            return True
    return False


def overlap_seconds(
    a_start: float, a_end: float, b_start: float, b_end: float
) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def normalize_joined_text(parts: list[str]) -> str:
    text = " ".join(part.strip() for part in parts if part and part.strip())
    text = MULTISPACE_RE.sub(" ", text).strip()
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return text


def ends_with_terminal_punctuation(text: str) -> bool:
    return bool(TERMINAL_PUNCTUATION_RE.search(text.strip()))


def compute_time_block(raw_segments: list[dict]) -> dict:
    start_seconds = raw_segments[0]["time"]["start_seconds"]
    end_seconds = raw_segments[-1]["time"]["end_seconds"]
    return {
        "start": raw_segments[0]["time"].get("start_ts"),
        "end": raw_segments[-1]["time"].get("end_ts"),
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "duration_seconds": end_seconds - start_seconds,
    }


def group_text(group: list[dict]) -> str:
    return normalize_joined_text([item["transcript_segment"]["text"] for item in group])


def count_words(text: str) -> int:
    return len(text.split()) if text else 0


def should_flush_on_punctuation(text: str, group_size: int) -> bool:
    if not ends_with_terminal_punctuation(text):
        return False

    enough_chars = len(text) >= SOFT_MIN_CHARS
    enough_words = count_words(text) >= SOFT_MIN_WORDS
    enough_segments = group_size >= SOFT_MIN_SOURCE_SEGMENTS
    return enough_segments and (enough_chars or enough_words)


def should_flush_on_max_size(text: str, duration_seconds: float) -> bool:
    return len(text) >= HARD_MAX_CHARS or duration_seconds >= HARD_MAX_DURATION_SECONDS


def attribute_transcript_segment(
    t_seg: dict,
    diarization_segments: list[dict],
    min_other_overlap_seconds: float,
    multiple_speaker_ratio_threshold: float,
) -> dict:
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
        overlaps_by_speaker[speaker_id] = overlaps_by_speaker.get(speaker_id, 0.0) + ov
        overlap_details.setdefault(speaker_id, []).append(d_seg["segment_id"])

    flags = SegmentFlag.NONE
    primary_speaker_id: str | None = None
    primary_confidence: float | None = None
    other_speakers: list[dict] = []
    provenance_diarization_ids: list[str] = []

    if not overlaps_by_speaker:
        flags |= SegmentFlag.NEEDS_REVIEW
    else:
        ranked = sorted(overlaps_by_speaker.items(), key=lambda x: x[1], reverse=True)
        primary_speaker_id, primary_overlap = ranked[0]
        primary_confidence = primary_overlap / t_duration if t_duration > 0 else None

        provenance_diarization_ids.extend(overlap_details.get(primary_speaker_id, []))

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
                provenance_diarization_ids.extend(overlap_details.get(speaker_id, []))

        if other_speakers:
            flags |= SegmentFlag.MULTIPLE_SPEAKERS

    if t_duration < 1.5:
        flags |= SegmentFlag.IS_SHORT

    return {
        "transcript_segment": t_seg,
        "primary_speaker_id": primary_speaker_id,
        "primary_confidence": primary_confidence,
        "flags": flags,
        "other_speakers": other_speakers,
        "provenance_diarization_ids": sorted(set(provenance_diarization_ids)),
    }


def build_final_segment(
    idx: int,
    group: list[dict],
    language: str | None,
    flush_reason: str,
) -> dict:
    raw_segments = [item["transcript_segment"] for item in group]
    time_block = compute_time_block(raw_segments)
    text_raw = normalize_joined_text([seg["text"] for seg in raw_segments])

    flags = SegmentFlag.NONE
    all_other_speakers: dict[str, dict] = {}
    diarization_ids: set[str] = set()
    transcript_ids: list[str] = []
    source_segments: list[dict] = []

    weighted_confidence_sum = 0.0
    weighted_confidence_duration = 0.0
    primary_speaker_id = group[0]["primary_speaker_id"]

    for item in group:
        seg = item["transcript_segment"]
        seg_duration = seg["time"].get("duration_seconds", 0.0) or 0.0
        transcript_ids.append(seg["segment_id"])
        diarization_ids.update(item["provenance_diarization_ids"])
        flags |= item["flags"]

        conf = item["primary_confidence"]
        if conf is not None and seg_duration > 0:
            weighted_confidence_sum += conf * seg_duration
            weighted_confidence_duration += seg_duration

        for other in item["other_speakers"]:
            current = all_other_speakers.get(other["speaker_id"])
            if current is None:
                all_other_speakers[other["speaker_id"]] = other.copy()
            else:
                current["overlap_seconds"] += other["overlap_seconds"]
                current["overlap_ratio"] += other["overlap_ratio"]

        source_segments.append(
            {
                "transcript_segment_id": seg["segment_id"],
                "time": seg["time"],
                "text": seg["text"],
                "primary_speaker_id": item["primary_speaker_id"],
                "primary_confidence": item["primary_confidence"],
                "diarization_segment_ids": item["provenance_diarization_ids"],
            }
        )

    primary_confidence = (
        weighted_confidence_sum / weighted_confidence_duration
        if weighted_confidence_duration > 0
        else None
    )

    other_speakers = sorted(
        all_other_speakers.values(),
        key=lambda x: (x["overlap_seconds"], x["speaker_id"]),
        reverse=True,
    )

    return {
        "segment_id": f"seg_{idx:06d}",
        "time": time_block,
        "speaker": {
            "speaker_id": primary_speaker_id,
            "speaker_label": None,
            "speaker_label_source": None,
            "confidence": primary_confidence,
        },
        "text": {
            "raw": text_raw,
            "normalized": None,
            "language": language,
        },
        "flags": int(flags),
        "other_speakers": other_speakers,
        "entities": [],
        "keywords": [],
        "provenance": {
            "transcript_segment_ids": transcript_ids,
            "diarization_segment_ids": sorted(diarization_ids),
            "source_segments": source_segments,
            "flush_reason": flush_reason,
            "stage_created_by": "merge",
        },
    }


def group_attributed_segments(attributed_segments: list[dict]) -> list[dict]:
    grouped: list[dict] = []
    current_group: list[dict] = []

    def flush_current(reason: str) -> None:
        nonlocal current_group
        if not current_group:
            return
        grouped.append({"group": current_group, "flush_reason": reason})
        current_group = []

    for item in attributed_segments:
        if not current_group:
            current_group.append(item)
            continue

            previous_item = current_group[-1]
            previous_speaker = previous_item["primary_speaker_id"]
            current_speaker = item["primary_speaker_id"]
            speaker_changed = previous_speaker != current_speaker

            previous_text = previous_item["transcript_segment"]["text"]
            previous_sentence_closed = ends_with_terminal_punctuation(previous_text)

            merge_veto_on_boundary_overlap = (
                MERGE_VETO_ON_PUNCTUATED_BOUNDARY_WITH_OVERLAP
                and previous_sentence_closed
                and has_meaningful_other_speaker_overlap(item)
            )

            if speaker_changed:
                flush_current("speaker_change")
                current_group.append(item)
                continue

            if merge_veto_on_boundary_overlap:
                flush_current("punctuated_boundary_with_overlap")
                current_group.append(item)
                continue

        current_group.append(item)

        current_text = group_text(current_group)
        current_duration = compute_time_block(
            [x["transcript_segment"] for x in current_group]
        )["duration_seconds"]

        if should_flush_on_max_size(current_text, current_duration):
            flush_current("max_size")
            continue

        if should_flush_on_punctuation(current_text, len(current_group)):
            flush_current("terminal_punctuation")

    flush_current("end_of_input")
    return grouped


def merge_segments(
    input_json_path: Path,
    output_json_path: Path | None = None,
    min_other_overlap_seconds: float = DEFAULT_MIN_OTHER_OVERLAP_SECONDS,
    multiple_speaker_ratio_threshold: float = DEFAULT_MULTIPLE_SPEAKER_RATIO_THRESHOLD,
) -> dict:
    input_json_path = input_json_path.resolve()

    if not input_json_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

    output_json_path = output_json_path or build_default_output_path(input_json_path)
    doc = load_document(input_json_path)

    transcript_segments = doc["transcript"]["raw_segments"]
    diarization_segments = doc["diarization"]["raw_segments"]
    language = doc["transcript"].get("language_detected")

    attributed_segments = [
        attribute_transcript_segment(
            t_seg=t_seg,
            diarization_segments=diarization_segments,
            min_other_overlap_seconds=min_other_overlap_seconds,
            multiple_speaker_ratio_threshold=multiple_speaker_ratio_threshold,
        )
        for t_seg in transcript_segments
    ]

    grouped_segments = group_attributed_segments(attributed_segments)
    final_segments = [
        build_final_segment(
            idx=i,
            group=entry["group"],
            language=language,
            flush_reason=entry["flush_reason"],
        )
        for i, entry in enumerate(grouped_segments, start=1)
    ]

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
        description=(
            "Merge transcript and diarization into speaker-attributed, sentence-aware segments."
        )
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
        default=DEFAULT_MIN_OTHER_OVERLAP_SECONDS,
        help="Minimum overlap in seconds to keep as other speaker",
    )
    parser.add_argument(
        "--multiple-speaker-ratio-threshold",
        type=float,
        default=DEFAULT_MULTIPLE_SPEAKER_RATIO_THRESHOLD,
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
