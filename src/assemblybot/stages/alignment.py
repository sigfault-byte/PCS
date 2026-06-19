import argparse
from pathlib import Path

from assemblybot.alignment_config import (
    AlignmentConfig,
    add_alignment_arguments,
)
from assemblybot.helper.directory import build_default_output_path
from assemblybot.helper.document import load_document, save_document
from assemblybot.models.alignment import TranscriptDiarizationMatch
from assemblybot.models.diarization import DiarizationRawSegment
from assemblybot.models.document import CanonicalDocument
from assemblybot.models.flags import SegmentFlag, has_flag
from assemblybot.models.time import TimeRange
from assemblybot.models.transcript import TranscriptRawSegment


def ranges_overlap_seconds(a: TimeRange, b: TimeRange) -> float:
    return max(
        0.0,
        min(a.end_seconds, b.end_seconds) - max(a.start_seconds, b.start_seconds),
    )


def calculate_speaker_evidence_score(
    cumulative_overlap: float,
    longest_overlap: float,
    overlap_segment_count: int,
    config: AlignmentConfig,
) -> float:
    raw_score = (
        cumulative_overlap
        + config.speaker_evidence_longest_overlap_weight * longest_overlap
        - config.speaker_evidence_extra_segment_penalty
        * max(0, overlap_segment_count - 1)
    )
    return max(0.0, raw_score)


def propagate_adjacent_transcript_anomaly_flags(
    transcript_segments: list[TranscriptRawSegment],
    config: AlignmentConfig,
) -> None:
    source_indexes = [
        index
        for index, segment in enumerate(transcript_segments)
        if any(
            has_flag(segment.flags, flag)
            for flag in config.anomaly_flags_to_propagate
        )
    ]

    for index in source_indexes:
        for neighbor_index in (index - 1, index + 1):
            if 0 <= neighbor_index < len(transcript_segments):
                transcript_segments[
                    neighbor_index
                ].flags |= SegmentFlag.ADJACENT_INFORMATION_RATE_ANOMALY


def build_transcript_diarization_match(
    transcript_segment: TranscriptRawSegment,
    diarization_segments: list[DiarizationRawSegment],
    config: AlignmentConfig,
) -> TranscriptDiarizationMatch:
    match = TranscriptDiarizationMatch(
        transcript_segment_id=transcript_segment.segment_id,
        flags=0,
    )

    for diarization_segment in diarization_segments:
        overlap_seconds = ranges_overlap_seconds(
            transcript_segment.time,
            diarization_segment.time,
        )
        if overlap_seconds <= 0.0:
            continue

        # TODO: find a weighted logic depending on the lenght of the segment, this is a bit too naive.
        speaker_id = diarization_segment.speaker_id
        match.diarization_segment_ids.append(diarization_segment.segment_id)
        match.total_overlap_seconds += overlap_seconds
        match.speaker_overlap_seconds[speaker_id] = (
            match.speaker_overlap_seconds.get(speaker_id, 0.0) + overlap_seconds
        )
        match.speaker_longest_overlap_seconds[speaker_id] = max(
            match.speaker_longest_overlap_seconds.get(speaker_id, 0.0),
            overlap_seconds,
        )
        match.speaker_overlap_segment_count[speaker_id] = (
            match.speaker_overlap_segment_count.get(speaker_id, 0) + 1
        )

    match.speaker_ids = sorted(match.speaker_overlap_seconds)

    for speaker_id in match.speaker_ids:
        cumulative_overlap = match.speaker_overlap_seconds[speaker_id]
        longest_overlap = match.speaker_longest_overlap_seconds[speaker_id]
        overlap_segment_count = match.speaker_overlap_segment_count[speaker_id]
        match.speaker_evidence_score[speaker_id] = calculate_speaker_evidence_score(
            cumulative_overlap=cumulative_overlap,
            longest_overlap=longest_overlap,
            overlap_segment_count=overlap_segment_count,
            config=config,
        )

    if match.speaker_evidence_score:
        winning_score = max(match.speaker_evidence_score.values())
        tied_speaker_ids = [
            speaker_id
            for speaker_id, score in match.speaker_evidence_score.items()
            if score == winning_score
        ]
        if len(tied_speaker_ids) > 1:
            match.flags |= SegmentFlag.TIE_BREAK_SPEAKER

        # Dirty tie break: alphabetical speaker id wins.
        match.probable_speaker_id = min(tied_speaker_ids)
        match.winning_evidence_score = match.speaker_evidence_score[
            match.probable_speaker_id
        ]
        match.winning_overlap_seconds = match.speaker_overlap_seconds[
            match.probable_speaker_id
        ]

        total_evidence_score = sum(match.speaker_evidence_score.values())
        if total_evidence_score:
            match.speaker_confidence = (
                match.winning_evidence_score / total_evidence_score
            )

    return match


def build_transcript_diarization_matches(
    document: CanonicalDocument,
    config: AlignmentConfig,
) -> list[TranscriptDiarizationMatch]:
    return [
        build_transcript_diarization_match(
            transcript_segment,
            document.diarization.raw_segments,
            config,
        )
        for transcript_segment in document.transcript.raw_segments
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Propagate neighboring transcript anomaly flags and align transcript "
            "segments with diarization speakers."
        )
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Existing canonical document JSON to update",
    )
    parser.add_argument(
        "--output-json",
        help="Optional output JSON path (default: generated in interim directory)",
    )
    add_alignment_arguments(parser)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AlignmentConfig.from_args(args)

    input_json_path = Path(args.input_json).resolve()
    document = load_document(input_json_path)

    output_json_path = (
        Path(args.output_json).resolve()
        if args.output_json
        else build_default_output_path(
            Path(document.source.input_path or input_json_path),
            "_03_alignment",
            "json",
        )
    )

    # STEP 1 -- Propagate adjacent transcript anomaly flags.
    propagate_adjacent_transcript_anomaly_flags(
        document.transcript.raw_segments,
        config,
    )

    # STEP 2 -- Build transcript diarization matches.
    document.alignment.transcript_diarization_matches = (
        build_transcript_diarization_matches(document, config)
    )

    # STEP 3 -- Save document.
    save_document(document, output_json_path)


if __name__ == "__main__":
    main()
