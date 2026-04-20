from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from assemblybot.helper.artifact import save_npz
from assemblybot.helper.directory import build_default_output_path
from assemblybot.helper.document import load_document, save_document
from assemblybot.models.diarization import (
    CollapsedDiarizationSegment,
    DiarizationArtifacts,
    DiarizationRawSegment,
    DiarizationSection,
)
from assemblybot.models.document import CanonicalDocument
from assemblybot.models.factories import (
    create_empty_document,
    mark_stage_completed,
    mark_stage_failed,
    mark_stage_running,
)
from assemblybot.models.final_segment import FinalSegment
from assemblybot.models.time import TimeRange
from assemblybot.models.transcript import TranscriptRawSegment

DIARIZATION_SEGMENT_GAP = 2.0


def collapse_diarization_segments(
    diarization: DiarizationSection,
) -> tuple[list[CollapsedDiarizationSegment], list[tuple[float, float]]]:

    raw_segments = diarization.raw_segments

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
        midpoint = (
            transcript_segment.time.start_seconds + transcript_segment.time.end_seconds
        ) / 2.0

        assigned = False
        for collapsed_segment in collapsed_sorted:
            if (
                collapsed_segment.time.start_seconds
                <= midpoint
                <= collapsed_segment.time.end_seconds
            ):
                assigned_segments[collapsed_segment.segment_id].append(
                    transcript_segment
                )
                assigned = True
                break

        if assigned:
            continue

        in_gap = False
        for gap_start, gap_end in true_gaps_sorted:
            if gap_start <= midpoint <= gap_end:
                transcript_ids_in_gap.add(transcript_segment.segment_id)
                in_gap = True
                break

        if in_gap:
            continue

        unmatched_transcript_ids.add(transcript_segment.segment_id)

    return assigned_segments, transcript_ids_in_gap, unmatched_transcript_ids


def merge(
    document: CanonicalDocument,
    output_json_path: Path,
) -> CanonicalDocument:

    transcript_seg = document.transcript.raw_segments
    collapse_diarization_seg, true_gaps = collapse_diarization_segments(
        document.diarization
    )
    assigned_segments, transcript_ids_in_gap, unmatched_transcript_ids = (
        assign_transcript_segments(
            transcript_seg,
            collapse_diarization_seg,
            true_gaps,
        )
    )
    document.diarization.collapsed_segments = collapse_diarization_seg

    # DEBUG ----------
    print(f"Collapsed diarization segments: {len(collapse_diarization_seg)}")
    for segment in collapse_diarization_seg:
        print(
            segment.segment_id,
            segment.speaker_id,
            segment.time.start_seconds,
            segment.time.end_seconds,
            segment.source_diarization_segment_ids,
        )
    print(f"True gaps (>2s): {len(true_gaps)}")
    for gap in true_gaps:
        print(gap)
    print(f"Assigned collapsed segments: {len(assigned_segments)}")
    print(f"Transcript segment ids in gap: {sorted(transcript_ids_in_gap)}")
    print(f"Unmatched transcript segment ids: {unmatched_transcript_ids}")

    for seg in document.transcript.raw_segments:
        if seg.segment_id in transcript_ids_in_gap:
            print(
                seg.segment_id,
                seg.time.start_seconds,
                seg.time.end_seconds,
                seg.raw_text,
            )

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
    output_json_path = build_default_output_path(
        fake_path,
        "_03_merge",
        "json",
    )

    print(output_json_path)
    merge(document, output_json_path)


if __name__ == "__main__":
    main()
