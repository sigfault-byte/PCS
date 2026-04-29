from __future__ import annotations

import argparse
from pathlib import Path

from assemblybot.helper.directory import build_default_output_path
from assemblybot.helper.document import load_document, save_document
from assemblybot.models.diarization import (
    CollapsedDiarizationSegment,
    DiarizationSection,
)
from assemblybot.models.document import CanonicalDocument
from assemblybot.models.factories import (
    mark_stage_completed,
    mark_stage_failed,
    mark_stage_running,
)
from assemblybot.models.final_segment import (
    FinalSegment,
    Provenance,
    SpeakerInfo,
    TextInfo,
)
from assemblybot.models.time import TimeRange
from assemblybot.models.transcript import TranscriptRawSegment
from src.assemblybot.models.flags import SegmentFlag

# Diarization max gap
DIARIZATION_SEGMENT_GAP = 4

# Pyannote tutorial-style fallback:
# if there is no overlap at all, assign to the nearest collapsed diarization
# segment only when the transcript midpoint is close enough.
NEAREST_ASSIGN_MAX_DISTANCE_SECONDS = 1.5


def compute_overlap_seconds(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def collapse_diarization_segments(
    diarization: DiarizationSection,
) -> tuple[list[CollapsedDiarizationSegment], list[tuple[float, float]]]:
    raw_segments = sorted(
        diarization.raw_segments,
        key=lambda segment: segment.time.start_seconds,
    )

    if not raw_segments:
        return [], []

    collapsed_segments: list[CollapsedDiarizationSegment] = []
    true_gaps: list[tuple[float, float]] = []
    counter = 1

    current = CollapsedDiarizationSegment(
        segment_id=f"cdia_{counter:06d}",
        time=TimeRange.from_seconds(
            raw_segments[0].time.start_seconds,
            raw_segments[0].time.end_seconds,
        ),
        speaker_id=raw_segments[0].speaker_id,
        source_diarization_segment_ids=[raw_segments[0].segment_id],
    )

    for segment in raw_segments[1:]:
        gap_seconds = segment.time.start_seconds - current.time.end_seconds

        should_merge = (
            segment.speaker_id == current.speaker_id
            and gap_seconds < DIARIZATION_SEGMENT_GAP
        )

        if should_merge:
            current.time = TimeRange.from_seconds(
                current.time.start_seconds,
                max(current.time.end_seconds, segment.time.end_seconds),
            )
            current.source_diarization_segment_ids.append(segment.segment_id)
            continue

        if gap_seconds >= DIARIZATION_SEGMENT_GAP:
            true_gaps.append((current.time.end_seconds, segment.time.start_seconds))

        collapsed_segments.append(current)
        counter += 1
        current = CollapsedDiarizationSegment(
            segment_id=f"cdia_{counter:06d}",
            time=TimeRange.from_seconds(
                segment.time.start_seconds,
                segment.time.end_seconds,
            ),
            speaker_id=segment.speaker_id,
            source_diarization_segment_ids=[segment.segment_id],
        )

    collapsed_segments.append(current)
    return collapsed_segments, true_gaps


def assign_transcript_segments(
    transcript_segments: list[TranscriptRawSegment],
    collapsed_segments: list[CollapsedDiarizationSegment],
    true_gaps: list[tuple[float, float]],
) -> tuple[
    dict[str, list[TranscriptRawSegment]],
    set[str],
    set[str],
]:
    """
    Pyannote-style merge:
    1. For each transcript segment, compute overlap with every collapsed diarization
       segment.
    2. Assign the transcript segment to the collapsed diarization segment with the
       largest overlap.
    3. If there is no overlap at all, fall back to the nearest collapsed diarization
       segment by midpoint, but only if it is close enough.
    4. If midpoint falls in a long true gap, classify it in the gap bucket.
    5. Otherwise keep it unmatched.

    This keeps the same return shape as the existing implementation so the rest of
    the file can stay unchanged.
    """
    assigned_segments: dict[str, list[TranscriptRawSegment]] = {
        segment.segment_id: [] for segment in collapsed_segments
    }
    transcript_ids_in_gap: set[str] = set()
    unmatched_transcript_ids: set[str] = set()

    collapsed_sorted = sorted(
        collapsed_segments,
        key=lambda segment: segment.time.start_seconds,
    )
    true_gaps_sorted = sorted(true_gaps)

    for transcript_segment in transcript_segments:
        transcript_start = transcript_segment.time.start_seconds
        transcript_end = transcript_segment.time.end_seconds
        midpoint = (transcript_start + transcript_end) / 2.0

        best_segment: CollapsedDiarizationSegment | None = None
        best_overlap = 0.0

        for collapsed_segment in collapsed_sorted:
            overlap = compute_overlap_seconds(
                transcript_start,
                transcript_end,
                collapsed_segment.time.start_seconds,
                collapsed_segment.time.end_seconds,
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_segment = collapsed_segment

        # Primary assignment: largest temporal overlap wins.
        if best_segment is not None and best_overlap > 0.0:
            assigned_segments[best_segment.segment_id].append(transcript_segment)
            continue

        # Long uncovered region: keep it in the explicit gap bucket.
        in_gap = False
        for gap_start, gap_end in true_gaps_sorted:
            if gap_start <= midpoint <= gap_end:
                transcript_ids_in_gap.add(transcript_segment.segment_id)
                in_gap = True
                break

        if in_gap:
            continue

        # No overlap at all: optional nearest-segment fallback by midpoint.
        nearest_segment: CollapsedDiarizationSegment | None = None
        nearest_distance = float("inf")

        for collapsed_segment in collapsed_sorted:
            if midpoint < collapsed_segment.time.start_seconds:
                distance = collapsed_segment.time.start_seconds - midpoint
            elif midpoint > collapsed_segment.time.end_seconds:
                distance = midpoint - collapsed_segment.time.end_seconds
            else:
                distance = 0.0

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_segment = collapsed_segment

        if (
            nearest_segment is not None
            and nearest_distance <= NEAREST_ASSIGN_MAX_DISTANCE_SECONDS
        ):
            assigned_segments[nearest_segment.segment_id].append(transcript_segment)
            continue

        unmatched_transcript_ids.add(transcript_segment.segment_id)

    return assigned_segments, transcript_ids_in_gap, unmatched_transcript_ids


def build_raw_text(transcript_segments: list[TranscriptRawSegment]) -> str:
    text_parts = [segment.raw_text.strip() for segment in transcript_segments]
    return " ".join(part for part in text_parts if part)


def build_transcript_token_bounds(
    transcript_segments: list[TranscriptRawSegment],
) -> tuple[int | None, int | None]:
    start_token_ids = [
        segment.start_token_id
        for segment in transcript_segments
        if segment.start_token_id is not None
    ]
    end_token_ids = [
        segment.end_token_id
        for segment in transcript_segments
        if segment.end_token_id is not None
    ]

    token_start_id = min(start_token_ids) if start_token_ids else None
    token_end_id = max(end_token_ids) if end_token_ids else None
    return token_start_id, token_end_id


def build_final_segments(
    document: CanonicalDocument,
    collapsed_segments: list[CollapsedDiarizationSegment],
    assigned_segments: dict[str, list[TranscriptRawSegment]],
) -> list[FinalSegment]:
    final_segments: list[FinalSegment] = []
    language_detected = document.transcript.language_detected

    collapsed_sorted = sorted(
        collapsed_segments,
        key=lambda segment: segment.time.start_seconds,
    )

    for idx, collapsed_segment in enumerate(collapsed_sorted, start=1):
        transcript_segments = sorted(
            assigned_segments.get(collapsed_segment.segment_id, []),
            key=lambda segment: segment.time.start_seconds,
        )
        transcript_token_start_id, transcript_token_end_id = (
            build_transcript_token_bounds(transcript_segments)
        )

        final_segments.append(
            FinalSegment(
                segment_id=f"seg_{idx:06d}",
                time=TimeRange.from_seconds(
                    collapsed_segment.time.start_seconds,
                    collapsed_segment.time.end_seconds,
                ),
                speaker=SpeakerInfo(
                    speaker_id=collapsed_segment.speaker_id,
                    speaker_label=None,
                    speaker_label_source=None,
                    confidence=None,
                ),
                text=TextInfo(
                    raw=build_raw_text(transcript_segments),
                    normalized=None,
                    language=language_detected,
                ),
                flags=SegmentFlag.NONE,
                entities=[],
                keywords=[],
                provenance=Provenance(
                    transcript_segment_ids=[
                        segment.segment_id for segment in transcript_segments
                    ],
                    diarization_segment_ids=list(
                        collapsed_segment.source_diarization_segment_ids
                    ),
                    transcript_token_start_id=transcript_token_start_id,
                    transcript_token_end_id=transcript_token_end_id,
                    stage_created_by="merge",
                ),
            )
        )

    return final_segments


def merge(
    document: CanonicalDocument,
    output_json_path: Path,
) -> CanonicalDocument:
    transcript_segments = document.transcript.raw_segments
    collapsed_diarization_segments, true_gaps = collapse_diarization_segments(
        document.diarization
    )
    assigned_segments, transcript_ids_in_gap, unmatched_transcript_ids = (
        assign_transcript_segments(
            transcript_segments,
            collapsed_diarization_segments,
            true_gaps,
        )
    )
    document.diarization.collapsed_segments = collapsed_diarization_segments
    document.segments = build_final_segments(
        document,
        collapsed_diarization_segments,
        assigned_segments,
    )

    orphan_transcript_ids = sorted(
        transcript_ids_in_gap.union(unmatched_transcript_ids)
    )
    print(f"Orphan transcript segment ids: {orphan_transcript_ids}")

    mark_stage_completed(
        document,
        "merge",
        output_path=str(output_json_path),
    )
    save_document(document, output_json_path)
    print(f"Saved merged document: {output_json_path}")

    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merging transcript segment to diarization segments and save in a new caonnonical document."
    )

    parser.add_argument(
        "--input-json",
        required=True,
        help="Optional existing canonical document JSON to resume from",
    )

    parser.add_argument(
        "--output-json",
        help="Optional output JSON path (default: generated in interim directory)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    document = load_document(Path(args.input_json).resolve())

    input_json_path = Path(args.input_json).resolve()
    clean_stem = input_json_path.stem.removesuffix("_02_transcription")
    fake_path = input_json_path.with_name(clean_stem + input_json_path.suffix)
    output_json_path = (
        Path(args.output_json).resolve()
        if args.output_json
        else build_default_output_path(
            fake_path,
            "_03_merge_pyannote_style",
            "json",
        )
    )

    mark_stage_running(document, "merge")

    try:
        merge(document, output_json_path)
    except Exception as exc:
        mark_stage_failed(document, "merge", str(exc))
        save_document(document, output_json_path)
        raise


if __name__ == "__main__":
    main()
